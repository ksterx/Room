from abc import ABC, abstractmethod

from room.train.memory import Memory


class Agent(ABC):
    def __init__(self, obs_space, act_space, cfg):
        self.obs_space = obs_space
        self.act_space = act_space
        self.cfg = cfg

    @abstractmethod
    def act(self, obs):
        pass

    def log(self):
        pass
