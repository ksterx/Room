from abc import ABC, abstractmethod
from typing import List, Optional, Union

import ray
import torch
from torch import optim

from room import notice
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.utils import get_optimizer, get_param
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory, memory_aliases


class Trainer(ABC):
    def __init__(
        self,
        env: EnvWrapper,
        agents: Union[Agent, List[Agent]],
        timesteps: Optional[int] = None,
        memory: Optional[Union[str, Memory]] = None,
        logger: Optional[Logger] = None,
        cfg: dict = None,
        callbacks: Union[Callback, List[Callback]] = None,
        optimizer: Optional[Union[str, optim.Optimizer]] = None,
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
        self.cfg = cfg
        self.callbacks = callbacks

        optimizer = get_optimizer(optimizer, cfg)

        if isinstance(agents, List[Agent]):
            for agent in self.agents:
                agent.optimizer = optimizer
        elif isinstance(agents, Agent):
            self.agents.optimizer = optimizer
        else:
            raise TypeError("agents should be either Agent or List[Agent]")

        self.agents = agents
        self.logger = get_param(logger, "logger", cfg)
        self.timesteps = get_param(timesteps, "timesteps", cfg)
        self.memory = get_param(memory, "memory", cfg, memory_aliases)  # TODO: Vecenv

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
