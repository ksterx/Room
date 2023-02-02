from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import ray
import torch
from kxmod.service import SlackBot
from torch import optim
from tqdm import trange

from room import notice
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.utils import get_device, get_param, is_debug
from room.envs.utils import get_action_shape, get_obs_shape
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory, registered_memories
from room.trainers import Trainer


class OffPolicyTrainer(Trainer):
    def __init__(
        self,
        memory: Optional[Union[str, Memory]] = None,
        capacity: Optional[int] = None,
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
    ) -> None:

        super().__init__(
            env_name=env_name,
            agent=agent,
            num_agents=num_agents,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            timesteps=timesteps,
            device=device,
            cfg=cfg,
            callbacks=callbacks,
            *args,
            **kwargs,
        )

        self.memory = get_param(memory, "memory", cfg, registered_memories, show=is_debug(cfg))
        self.capacity = get_param(capacity, "capacity", cfg, show=is_debug(cfg))
        if isinstance(memory, str):
            self.memory = self.memory(capacity=self.capacity, device=self.device)

    def train(self):

        self.on_train_start()

        states = ray.get([agent.reset_env.remote() for agent in self.agents])

        for t in trange(self.timesteps):

            self.on_timestep_start(t)

            actions = ray.get([agent.act.remote(states) for agent in self.agents])

    def eval(self):
        pass
