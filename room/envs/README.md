# Environment

## Usage

```python
import gym

from room.envs import GymEnvWrapper


env = gym.make("CartPole-v1", render_mode="human")
wrapped = GymEnvWrapper(env)
terminated = False
wrapped.reset()  # 'reset' method must be called before step

while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, _, _ = wrapped.step(action)
    print(obs, reward, terminated)
    wrapped.render()
```
