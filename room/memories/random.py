import random
from typing import List, Optional, Union

import torch

from room.memories.base import Memory


class RandomMemory(Memory):
    def __init__(self, capacity: int, device: Optional[Union[str, torch.device]] = None):
        super().__init__(capacity=capacity, device=device)

    def __str__(self):
        return f"{self.__class__.__name__} (capacity: {self.capacity})"

    def add(self, experience, *args):
        self.experiences.append(experience)

    def sample(self, batch_size):
        super().sample(batch_size)
        batch = random.sample(self.experiences, batch_size)
        return self.sort_by_key(batch)
