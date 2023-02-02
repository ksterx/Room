# Room

This is a reinforcement learning library.

## Installation

### Prerequisites

- gym > 0.26
- Ubuntu 22.04

### Using PyPI

Run the following command at top-level directory:

```bash
pip install -e .
```

### Using Docker

Coming soon

## Usage

- **Train an agent**

  ```bash
  python experiments/<TASK NAME>/train.py <OPTIONS>
  ```

- **View training progress and results with MLFlow (default: localhost:5000)**

  ```bash
  bash dashboard.sh
  ```

### Example

```python
python experiments/cartpole/train_dqn.py timesteps=100000 batch_size=32 agent.epsilon=0.1
```

## References

- SKRL  [[GitHub]](https://github.com/Toni-SM/skrl) [[Docs]](https://skrl.readthedocs.io/en/latest/index.html)

- Stable Baselines3  [[GitHub]](https://github.com/DLR-RM/stable-baselines3) [[Docs]](https://stable-baselines3.readthedocs.io/en/master/index.html)

- RL Games  [[GitHub]](https://github.com/Denys88/rl_games/tree/d6ccfa59c85865bc04d80ca56b3b0276fec82f90)

- Isaac Gym  [[Website]](https://developer.nvidia.com/issac-gym)
