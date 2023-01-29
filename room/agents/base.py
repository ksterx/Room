from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import gym
import ray
import torch
from omegaconf import DictConfig

from room import notice
from room.agents.policies import Policy, policies
from room.common.utils import get_device, get_optimizer, get_param
from room.memories.base import Memory
from room.networks import registered_models


class Agent(ABC):
    def __init__(
        self,
        model: Optional[Union[Policy, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Dict] = None,
        logger: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        self.model = get_param(model, "model", cfg)
        self.device = get_device(device)
        self.cfg = cfg
        self.logger = logger

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

    def on_before_step(self, timestep):
        pass

    def on_after_step(self):
        pass

    @abstractmethod
    def on_before_train(self, state_dim: Optional[int] = None, action_dim: Optional[int] = None):
        if isinstance(self.model, str):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.model = self._build_registered_model(model_name=self.model, state_dim=state_dim, action_dim=action_dim)

    def save(self, path):
        self.model.save(path)

    def load(self):
        pass

    def configure_optimizer(self, optimizer: Union[str, torch.optim.Optimizer], lr: Optional[float] = None):
        lr = get_param(lr, "lr", self.cfg)
        print(optimizer)
        self.optimizer = get_optimizer(optimizer, self.cfg)(self.model.parameters(), lr=lr)

    def _build_registered_model(self, model_name, state_dim, action_dim):
        self.model = registered_models[model_name].get(state_dim, action_dim)
