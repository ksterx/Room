import gym

from room.envs import GymEnvWrapper


def test_gym():
    env = gym.make("CartPole-v1", render_mode="human")
    wrapped = GymEnvWrapper(env)
    terminated = False
    wrapped.reset()
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = wrapped.step(action)
        print(obs, reward, terminated)
        wrapped.render()
