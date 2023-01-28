from abc import ABC, abstractmethod
from collections import deque
from typing import Union

import gym
import numpy as np
import torch

from room import notice
from room.common.preprocessing import get_space_shape


class Memory(ABC):
    def __init__(
        self,
        capacity: int,
        num_envs: int = 1,
    ):
        super().__init__()

        self.experiences = deque(maxlen=capacity)
        self.capacity = capacity
        self.num_envs = num_envs
        self.clear()

        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")

    def __len__(self):
        return len(self.experiences)

    def __str__(self):
        return f"{self.__class__.__name__} (capacity: {self.capacity})"

    def add(self, item):
        self.experiences.append(item)

    def add_batch(self):
        pass

    def sample(self):
        if len(self.experiences) == 0:
            notice.warning("Memory is empty")
            return None

    def sample_all(self):
        pass

    def clear(self):
        self.is_full = False
        self.memory_idx = 0

    @property
    def size(self):
        if self.is_full:
            return self.capacity
        return self.memory_idx
