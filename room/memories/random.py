import random

from room.memories.base import Memory


class RandomMemory(Memory):
    def __init__(self, capacity):
        super().__init__(capacity)

    def __str__(self):
        return f"{self.__class__.__name__} (capacity: {self.capacity})"

    def add(self, experience, *args):
        self.experiences.append(experience)

    def sample(self, batch_size):
        super().sample(batch_size)
        batch = random.sample(self.experiences, batch_size)
        return self.normalize_batch(batch)

    def update(self, idx, error):
        pass

    def randomize(self):
        random_order_experiences = self.experiences.copy()
        random.shuffle(random_order_experiences)
        return random_order_experiences
