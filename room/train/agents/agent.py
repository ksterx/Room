from abc import ABC, abstractmethod

from room.train.agents.memory import Memory, RolloutMemory


class Agent(ABC):
    def __init__(self, policy, obs_space, act_space, cfg, *args, **kwargs):
        self.policy = policy
        self.obs_space = obs_space
        self.act_space = act_space
        self.cfg = cfg

    @abstractmethod
    def act(self, obss):
        pass


class OnPolicyAgent(Agent):
    def __init__(self, policy, obs_space, act_space, cfg):
        super().__init__(policy, obs_space, act_space, cfg)
        self.memory = RolloutMemory(cfg.memory_size, cfg.num_envs)

    @abstractmethod
    def act(self, obs):
        pass

    @abstractmethod
    def update(self):
        pass

    def log(self):
        pass


class OffPolicyAgent(Agent):
    def __init__(self, obs_space, act_space, cfg):
        super().__init__(obs_space, act_space, cfg)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def log(self):
        pass
