from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init(self, env, agents: tuple | list, cfg):
        self.env = env
        self.agents = agents
        self.cfg = cfg

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def save(self):
        pass
