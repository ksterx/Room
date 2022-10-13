from abc import ABC, abstractmethod
from typing import List, Union

from room.envs.wrappers import EnvWrapper

from ..agents import Agent


class Trainer(ABC):
    def __init(self, env: EnvWrapper, agents: Union[Agent, List[Agent]], cfg):
        """Reinforcement learning trainer.

        Args:
            env (Env): environment to train on
            agents (Union[Agent, List[Agent]]): Agent(s) to train. If agents are being trained simultaneously, they should be passed as a list.
            cfg (yaml): Configurations
        """
        self.env = env
        self.agents = agents
        self.cfg = cfg

        # Set up agents
        self.num_agents = 0

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def save(self):
        pass
