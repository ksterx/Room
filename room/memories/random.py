from room.memories.base import Memory


class RandomMemory(Memory):
    def __init__(self, capacity):
        super().__init__(capacity)

    def add(self, *args):
        pass

    def sample(self, batch_size):
        pass

    def update(self, idx, error):
        pass

    def __len__(self):
        return 0

    def __str__(self):
        return f"RandomMemory(capacity={self.capacity})"

    def __repr__(self):
        return self.__str__()
