from room.memories.base import Memory, RolloutMemory
from room.memories.prioritized import PrioritizedReplayMemory
from room.memories.random import RandomMemory

memory_aliases = {
    "rollout": RolloutMemory,
    "memory": Memory,
    "random": RandomMemory,
    "prioritized": PrioritizedReplayMemory,
}
