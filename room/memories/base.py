from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional, Union

import gym
import numpy as np
import torch

from room import notice
from room.common.preprocessing import get_space_shape
from room.common.utils import get_device


class Memory(ABC):
    def __init__(self, capacity: int, device: Optional[Union[str, torch.device]] = None):
        super().__init__()

        self.experiences = deque(maxlen=capacity)
        self.capacity = capacity
        self.device = get_device(device)
        # self.clear()

        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")

    def __len__(self):
        return len(self.experiences)

    def __str__(self):
        return f"{self.__class__.__name__} (capacity: {self.capacity})"

    def add(self, experience: dict):
        """Add experience to memory.

        Args:
            item (dict): Experience dictionary.

                {"key": torch.Tensor(shape=(num_envs, *shape))}
        """
        self.experiences.append(experience)

    @abstractmethod
    def sample(self, batch_size: int):
        if len(self.experiences) == 0:
            notice.warning("Memory is empty")
            return None

    def sample_all(self):
        pass

    # def clear(self):
    #     self.is_full = False
    #     self.memory_idx = 0

    def is_full(self):
        return len(self.experiences) == self.capacity

    def sort_by_key(self, batch: List[Dict[str, torch.Tensor]]):
        """Normalize batch of experiences.

        Args:
            batch (List[Dict[str, torch.Tensor]]): Batch of experiences.

        Returns:
            Dict[str, torch.Tensor]: Normalized batch of experiences.
        """

        # [{key1: value11, key2: value21}, {key1: value12, key2: value22}] -> {key1: [value11, value12], key2: [value21, value22]}
        batch = {key: [item[key] for item in batch] for key in batch[0].keys()}
        for key, value in batch.items():
            if isinstance(value[0], torch.Tensor):
                batch[key] = torch.stack(value).squeeze(1).to(self.device)
        return batch
