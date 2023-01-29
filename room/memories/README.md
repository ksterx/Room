# `Memory` (Replay Buffer)

## API

`__init__`:

- `capacity` (int): The maximum number of transitions to store in the buffer. When the buffer overflows the old memories are dropped
<!-- - num_envs (int): The number of environments that will be interacting with the buffer. -->

`add`: Add a transition to the buffer

`sample`: Sample a batch of transitions from the buffer

## Usage

```python
from room.memories import RandomMemory
from room.trainers import Trainer

# Use a custom memory
memory = RandomMemory(capacity=1000)
trainer = Trainer(memory=memory)

# or use a registered memory by name in 'room.memories.registered_memories'
trainer = Trainer(memory="random")
```
