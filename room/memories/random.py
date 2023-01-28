import random

from room.memories.base import Memory


class RandomMemory(Memory):
    def __init__(self, capacity):
        super().__init__(capacity)

    def __str__(self):
        return f"{self.__class__.__name__} (capacity: {self.capacity})"

    def add(self, item, *args):
        self.experiences.append(item)

    def sample(self, batch_size):
        pass

    def update(self, idx, error):
        pass



    def randomize(self):
        random_order_experiences = self.experiences.copy()
        random.shuffle(random_order_experiences)
        return random_order_experiences
