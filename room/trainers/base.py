from abc import ABC, abstractmethod
from typing import List, Optional, Union

from room import log
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.config import set_param
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
        self.timesteps = set_param("timesteps", timesteps, cfg)
        if memory is None:
            self.memory = Memory  # TODO: get corresponding memory to the agent
        elif isinstance(memory, str):
            self.memory = memory_aliases[memory]
        self.memory = memory
        self.logger = logger
        self.cfg = cfg
        self.callbacks = callbacks

    @property
    def num_agents(self) -> int:
        if type(self.agents) in [tuple, list]:
            return len(self.agents)
        else:
            return 1

    @abstractmethod
    def train(self) -> None:
        """Execute training loop

        This method should be called before training loop is executed in the subclass.
        """

        if not self.logger:
            log.warning("No logger is set")

    @abstractmethod
    def eval(self):

        if not self.logger:
            log.warning("No logger is set")

        log.warning("Use SequentialTrainer or ParallelTrainer instead of Trainer")
        quit()

    @abstractmethod
    def save(self):
        log.warning("Use SequentialTrainer or ParallelTrainer instead of Trainer")
        quit()
