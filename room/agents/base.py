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
        optimizer: Union[torch.optim.Optimizer, str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Dict] = None,
        logger: Optional[Dict] = None,
        lr: Optional[float] = None,
        *args,
        **kwargs,
    ):
        self.model = get_param(model, "model", cfg)
        self.optimizer = optimizer
        self.device = get_device(device)
        self.cfg = cfg
        self.logger = logger
        self.lr = lr

    def initialize(self, state_shape: Optional[int] = None, action_shape: Optional[int] = None):
        self.state_shape = state_shape
        self.action_shape = action_shape
        if isinstance(self.model, str):
            self._build_registered_model(model_name=self.model, state_shape=state_shape, action_shape=action_shape)
        self.configure_optimizer(self.optimizer, lr=self.lr)

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

    def save(self, path):
        self.model.save(path)

    def load(self):
        pass

    def on_before_step(self):
        pass

    def on_after_step(self):
        pass

    def on_before_train(self):
        pass

    def configure_optimizer(self, optimizer: Union[str, torch.optim.Optimizer], lr: Optional[float] = None):
        lr = get_param(lr, "lr", self.cfg)
        self.optimizer = get_optimizer(optimizer, self.cfg)(self.model.parameters(), lr=lr)

    def _build_registered_model(self, model_name, state_shape, action_shape):
        self.model = registered_models[model_name](state_shape, action_shape)
