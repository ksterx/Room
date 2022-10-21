from abc import ABC, abstractmethod

import torch


class Memory(ABC):
    def __init__(self, memory_size, num_envs, obs_shape, action_shape):
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.reset()

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
    def __init__(self, memory_size: int, num_envs: int, obs_space, action_space):
        """Store transitions in memory for on-policy training (Advantage Actor-Critic)

        Args:
            memory_size (int): The number of steps for calculating the sum of rewards
            num_envs (int): The number of environments
        """
        super().__init__(memory_size, num_envs, obs_space, action_space)

    def reset(self) -> None:

        # TODO: torch.zeros
        self.observations = torch.zeros(self.memory_size, self.num_envs, *self.obs_space.shape)
        self.actions = torch.zeros(self.memory_size, self.num_envs, )
        self.rewards = torch.zeros(self.memory_size, self.num_envs)
        self.returns = torch.zeros(self.memory_size, self.num_envs)
        super.reset()

    def add(self, obs, action, reward, done, log_prob):

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
        assert self.is_full, "'get' method can be called only when memory is full"
        i = 0
        while i < self.memory_size * self.num_envs:
            yield self.actions[i], self.rewards[i], self.dones[i], self.infos[i]
            i += 1

    def estimate_advantages_and_returns(rewards, dones, values, gamma, gae_lambda, normalize_advantage, ):
        """


        Args:
            rewards: A batch of rewards
            dones: A batch of done flags
            values: A batch of values

        Returns:
            returns
            advantages
            values
        """
        advantages = torch.zeros_like(rewards)

        for step in reversed(range(self.memory_size)):
            if step == self.memory_size - 1:
                next_non_terminal = 1.0 - dones[step]
                next_value = next_values[step]
            else:
                pass

            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            advantage = delta + gamma * gae_lambda *

        returns = advantages + values
        if normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages, values


class RandomMemory(Memory):
    def __init__(self):
        NotImplemented


class PrioritizedMemory(Memory):
    def __init__(self):
        NotImplemented
