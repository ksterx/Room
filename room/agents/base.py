from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import gym
import ray
import torch
from omegaconf import DictConfig

from room import notice
from room.agents.policies import Policy, policies
from room.memories.base import Memory


def wrap_param(cfg: DictConfig, param, param_name: str):
    # Read param from config
    if param is None:
        param = cfg.agent[param_name]
    # Override with an function argument
    else:
        cfg.agent[param_name] = param

    return param, cfg


class Agent(ABC):
    def __init__(
        self,
        policy: Union[Policy, str],
        cfg: Optional[Dict] = None,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        self.cfg = cfg

        if isinstance(policy, str):
            self.policy = policies[policy]
        elif isinstance(policy, Policy):
            self.policy = policy
        else:
            notice.warning(f"Policy {policy} is not supported")

    @abstractmethod
    def act(self, obss):
        pass

    @abstractmethod
    def learn(self):
        pass

    def play(self, env, num_eps=1, render=True):
        if isinstance(env, gym.Env):
            for ep in range(num_eps):
                obs = env.reset()
                terminated = False
                ep_reward = 0
                while not terminated:
                    if render:
                        env.render()
                    action = self.act(obs)
                    obs, reward, terminated, _, _ = env.step(action)
                    ep_reward += reward

    def collect(self, transition):
        pass

    def on_before_step(self, timestep):
        pass

    def on_after_step(self):
        pass

    def save(self, path):
        self.policy.save(path)

    def load(self):
        pass
