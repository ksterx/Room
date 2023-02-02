from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import ray
import torch
from kxmod.service import SlackBot
from torch import optim

from room import notice
from room.agents import Agent, registered_agents
from room.common.callbacks import Callback
from room.common.utils import get_device, get_param, is_debug
from room.envs.utils import get_action_shape, get_obs_shape
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory, registered_memories
from room.networks import registered_models


class Trainer(ABC):
    def __init__(
        self,
        env_name: Optional[str] = None,
        agent: Optional[Agent] = None,
        num_agents: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[Union[str, optim.Optimizer]] = None,
        batch_size: Optional[int] = None,
        timesteps: Optional[int] = None,
        device: Optional[Union[str, int, List[int], torch.device, List[torch.device]]] = None,
        cfg: Optional[dict] = None,
        callbacks: Union[Callback, List[Callback]] = None,
        *args,
        **kwargs,
    ):
        """Reinforcement learning trainer.

        Args:
            env (Env): environment to train on
            agents (Union[Agent, List[Agent]]): Agent(s) to train.
                If agents are being trained simultaneously, they should be passed as a list.
            cfg (yaml): Configurations
        """
        self.env_name = get_param(env_name, "exp_name", cfg, show=is_debug(cfg))
        self.agent = get_param(agent, "agent_name", cfg, registered_agents, show=is_debug(cfg))
        if isinstance(self.agent, str):
            self.agent = registered_agents[self.agent]
        self.num_agents = get_param(num_agents, "num_agents", cfg, show=is_debug(cfg))
        self.model = model
        self.optimizer = optimizer
        self.batch_size = get_param(batch_size, "batch_size", cfg, show=is_debug(cfg))
        self.timesteps = get_param(timesteps, "timesteps", cfg, show=is_debug(cfg))
        self.device = device
        self.cfg = cfg
        self.callbacks = callbacks

        self.agents = self._get_agents()

        self._on_trainer_init()

        for i, agent in enumerate(self.agents):
            mo = ray.get(agent.get_model.remote())
            print(f"\nAgent {i}: \n{ray.get(agent.get_model.remote())}")
            notice.info(f"\nAgent {i}: \n{ray.get(agent.get_model.remote())}")

    @abstractmethod
    def train(self) -> None:
        return NotImplementedError

    @abstractmethod
    def eval(self):
        return NotImplementedError

    def on_timestep_start(self):
        self._loop_callback_agent("on_timestep_start")

    def on_timestep_end(self):
        self._loop_callback_agent("on_timestep_end")

    def on_episode_start(self):
        self._loop_callback_agent("on_episode_start")

    def on_episode_end(self, *args, **kwargs):
        self._loop_callback_agent("on_episode_end", *args, **kwargs)

    def on_train_start(self):
        self._loop_callback_agent("on_train_start")
        notice.info("Starting training...")

    def on_train_end(self):
        self._loop_callback_agent("on_train_end")
        msg = "Training finished!"
        SlackBot().say(msg)
        notice.info(msg)

    def _on_trainer_init(self):
        pass

    def _loop_callback_agent(self, method_name: str, *args, **kwargs):
        if self.callbacks is not None:
            for callback in self.callbacks:
                getattr(callback, method_name)(*args, **kwargs)
        if isinstance(self.agents, list):
            for agent in self.agents:
                getattr(agent, method_name)()
        elif isinstance(self.agents, Agent):
            getattr(self.agents, method_name)()

    def _get_agents(self, **kwargs):
        try:
            if len(self.device) == self.num_agents:
                agents = []
                for i, d in enumerate(self.device):
                    d = get_device(d)
                    agents.append(
                        self.agent.remote(
                            env=self.env_name,
                            model=self.model,
                            optimizer=self.optimizer,
                            device=d,
                            id=i,
                        )
                    )
            else:
                raise NotImplementedError("Device count is not equal to agent count")
        except TypeError:
            agents = [
                self.agent.remote(
                    env=self.env_name,
                    model=self.model,
                    optimizer=self.optimizer,
                    device=self.device,
                    id=i,
                    cfg=self.cfg,
                    callbacks=self.callbacks,
                    **kwargs,
                )
                for i in range(self.num_agents)
            ]
        return agents
