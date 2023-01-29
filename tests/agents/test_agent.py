import time
from typing import Dict, List, Optional, Union

import gym
import torch
from tqdm import trange

from room import notice
from room.agents import DQN, Agent
from room.agents.policies import Policy
from room.common.callbacks import Callback
from room.common.typing import CfgType
from room.common.utils import get_param
from room.envs import GymEnvWrapper
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory, RandomMemory
from room.networks.blocks import MLP
from room.trainers.base import Trainer

env = gym.make("CartPole-v1", render_mode="human")
env = GymEnvWrapper(env)
agent = DQN(model=[], optimizer=torch.optim.Adam, device="cuda", cfg={}, epsilon=0.1, gamma=0.01, lr=1e-3)
memory = RandomMemory(capacity=10)


def test_dqn():
    DQN(model=[], optimizer=torch.optim.Adam, device="cuda", cfg={}, epsilon=0.1, gamma=0.01, lr=1e-3)


def test_train():
    state = env.reset()
    state = state[0]

    for t in trange(1000):
        with torch.no_grad():

            action = agent.act(state[0])

        next_state, reward, terminated, truncated, info = env.step(action)
        memory.add(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )

        if not memory.is_full():
            continue
        else:
            batch = memory.sample(batch_size=3)
            agent.learn(batch)

        with torch.no_grad():
            if terminated.any() or truncated.any():
                states, infos = env.reset()

    agent.play(env)
