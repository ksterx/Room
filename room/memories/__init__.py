from room.memories.base import Memory
from room.memories.prioritized import PrioritizedMemory
from room.memories.random import RandomMemory

registered_memories = {
    "memory": Memory,
    "random": RandomMemory,
    "prioritized": PrioritizedMemory,
}
