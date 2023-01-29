import gym

from room.agents.policies import EpsilonGreedyPolicy, Policy
from room.memories.base import Memory, RolloutMemory

env = gym.make("CartPole-v1")
obs_space = env.observation_space.shape
action_space = env.action_space.n


def test_train(obs_space, action_space):
    memory = RolloutMemory(10, num_envs=1)
    policy = EpsilonGreedyPolicy(obs_space, epsilon=0.1)
    assert isinstance(memory, Memory)
    assert isinstance(policy, Policy)
    assert isinstance(policy, EpsilonGreedyPolicy)

    for ep in range(10):
        obs = env.reset()
        done = False
        while not done:
            action = policy.predict(obs)
            next_obs, reward, done, _, _ = env.step(action)
            memory.add(obs, action, reward, next_obs, done)
            obs = next_obs
