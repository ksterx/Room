# `Trainer`

`Trainer` is a class that provides a high-level API for training a model.

## API

`__init__`:

|parameter|description|
|---|---|
|`env` (`EnvWrapper`)| The environment to train on |
|`agents` (`Agent` or list of `Agent`)| The agent(s) to train |
|`memory` (`Memory`) | The memory to use for training |
|`optimizer` (`Optimizer`) | The optimizer to use for training

# `SequentialTrainer(Trainer)`

# `ParallelTrainer(Trainer)`

# `OffPolicyTrainer(Trainer)`
