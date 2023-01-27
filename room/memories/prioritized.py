from room.memories.base import Memory


class PrioritizedReplayMemory(Memory):
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.tree = SumTree(capacity)

    def add(self, *args):
        idx = self.tree.add(self._max_priority, *args)
        self._max_priority = max(self._max_priority, self.tree.data[idx])

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
        self._max_priority = max(self._max_priority, p)

    def _get_priority(self, error):
        return (np.abs(error) + self._epsilon) ** self.alpha

    def __len__(self):
        return len(self.tree)

    def __str__(self):
        return f"PrioritizedReplayMemory(capacity={self.capacity}, alpha={self.alpha}, beta={self.beta}, beta_increment={self.beta_increment})"

    def __repr__(self):
        return self.__str__()
