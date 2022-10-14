from abc import ABC, abstractmethod

import torch


class Memory(ABC):
    def __init__(self, memory_size, num_envs):
        self.memory_size = memory_size
        self.num_envs = num_envs

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def reset(self):
        self.is_full = False
        self.memory_idx = 0

    @property
    def size(self):
        if self.is_full:
            return self.memory_size
        return self.memory_idx


class RolloutMemory(Memory):
    def __init__(self, memory_size, num_envs):
        super().__init__(memory_size, num_envs)

    def reset(self) -> None:

        # TODO: torch.zeros
        self.observations = None
        self.actions = None
        self.rewards = torch.zeros(self.memory_size, self.num_envs)

    def add(self, obs, next_obs, action, reward, done, infos):

        self.observations[self.memory_idx] = obs.clone()
        self.actions[self.memory_idx] = action.clone()
        self.rewards[self.memory_idx] = reward.clone()

        self.memory_idx += 1
        assert self.memory_idx <= self.memory_size
        if self.memory_idx == self.memory_size:
            self.is_full = True

    def get(self):
        """Get a batch of data from memory

        Yields:
            data: A batch of transitions
        """
        i = 0
        while i < self.memory_size * self.num_envs:
            yield self.actions[i], self.rewards[i], self.dones[i], self.infos[i]
            i += 1


class RandomMemory(Memory):
    def __init__(self):
        NotImplemented


class PrioritizedMemory(Memory):
    def __init__(self):
        NotImplemented
