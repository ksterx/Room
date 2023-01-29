from abc import ABC, abstractmethod
from typing import List, Optional, Union

import ray
import torch
from torch import optim

from room import notice
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.utils import get_device, get_optimizer, get_param
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory, memory_aliases


class Trainer(ABC):
    def __init__(
        self,
        env: EnvWrapper,
        agents: Union[Agent, List[Agent]],
        memory: Optional[Union[str, Memory]] = None,
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
        self.agents = agents
        self.optimizer = optimizer
        self.device = get_device(device)
        self.cfg = cfg
        self.callbacks = callbacks
        self.logger = logger
        self.timesteps = get_param(timesteps, "timesteps", cfg)
        self.memory = get_param(memory, "memory", cfg, memory_aliases)  # TODO: Vecenv
        if isinstance(memory, str):
            self.memory = self.memory(capacity=kwargs["capacity"], device=self.device)

        if logger is None:
            notice.warning("No logger is set")

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

    def on_before_train(self):
        if isinstance(self.agents, list):
            for agent in self.agents:
                agent.optimizer = self.optimizer
                agent.on_before_train()
        elif isinstance(self.agents, Agent):
            self.agents.optimizer = self.optimizer
            self.agents.on_before_train()
        else:
            raise TypeError("agents should be either Agent or List[Agent]")

        for callback in self.callbacks:
            callback.on_before_train()
