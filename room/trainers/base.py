from abc import ABC, abstractmethod
from typing import List, Optional, Union

import ray
import torch
from kxmod.service import SlackBot
from torch import optim

from room import notice
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.utils import get_device, get_optimizer, get_param
from room.envs.utils import get_action_shape, get_obs_shape
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger, MLFlowLogger
from room.memories import Memory, registered_memories


class Trainer(ABC):
    def __init__(
        self,
        env: EnvWrapper,
        agents: Union[Agent, List[Agent]],
        memory: Optional[Union[str, Memory]] = None,
        batch_size: Optional[int] = None,
        optimizer: Optional[Union[str, optim.Optimizer]] = None,
        timesteps: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Logger] = None,
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
        self.env = env
        self.agents = agents if isinstance(agents, list) else [agents]
        self.batch_size = get_param(batch_size, "batch_size", cfg)
        self.optimizer = optimizer
        self.device = get_device(device)
        self.cfg = cfg
        self.callbacks = callbacks
        self.logger = logger
        self.timesteps = get_param(timesteps, "timesteps", cfg)
        self.memory = get_param(memory, "memory", cfg, registered_memories)  # TODO: Vecenv
        if isinstance(memory, str):
            self.memory = self.memory(capacity=kwargs["capacity"], device=self.device)

        self.obs_shape = get_obs_shape(env.observation_space)
        self.action_shape = get_action_shape(env.action_space)

        self._on_trainer_init(self.obs_shape, self.action_shape)

        for i, agent in enumerate(self.agents):
            notice.info(f"\nAgent {i}: \n{agent.model}")

    @property
    def num_agents(self) -> int:
        if type(self.agents) in [tuple, list]:
            return len(self.agents)
        else:
            return 1

    @abstractmethod
    def train(self) -> None:
        return NotImplementedError

    @abstractmethod
    def eval(self):
        return NotImplementedError

    @abstractmethod
    def save(self):
        notice.warning("Use SequentialTrainer or ParallelTrainer instead of Trainer")
        quit()

    def on_timestep_start(self):
        self._loop_callback_agent("on_timestep_start")

    def on_timestep_end(self):
        self._loop_callback_agent("on_timestep_end")

    def on_episode_start(self):
        self._loop_callback_agent("on_episode_start")

    def on_episode_end(self, metrics, episode):
        self._loop_callback_agent("on_episode_end", metrics, episode)

    def on_train_start(self):
        self._loop_callback_agent("on_train_start")

    def on_train_end(self):
        self._loop_callback_agent("on_train_end")

    def _on_trainer_init(self, state_shape, action_shape):
        if isinstance(self.agents, list):
            for agent in self.agents:
                agent.optimizer = self.optimizer
                agent.initialize(state_shape=state_shape, action_shape=action_shape)
        elif isinstance(self.agents, Agent):
            self.agents.optimizer = self.optimizer
            self.agents.initialize(state_shape=state_shape, action_shape=action_shape)
        else:
            raise TypeError("agents should be either Agent or List[Agent]")

    def _loop_callback_agent(self, method_name: str, *args, **kwargs):
        if self.callbacks is not None:
            for callback in self.callbacks:
                if isinstance(callback, Logger):
                    getattr(callback, method_name)(*args, **kwargs)
                else:
                    getattr(callback, method_name)()
        if isinstance(self.agents, list):
            for agent in self.agents:
                getattr(agent, method_name)()
        elif isinstance(self.agents, Agent):
            getattr(self.agents, method_name)()
