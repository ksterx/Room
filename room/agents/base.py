from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import gym
import ray
import torch
from gym.wrappers import RecordVideo
from omegaconf import DictConfig

from room import notice
from room.agents.policies import Policy, policies
from room.common.utils import get_device, get_optimizer, get_param, is_debug
from room.envs import register_env
from room.envs.wrappers import EnvWrapper
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

        self.model = get_param(model, "model", cfg, show=is_debug(cfg))
        self.optimizer = optimizer
        self.device = get_device(device)
        self.cfg = cfg
        self.logger = logger
        self.lr = lr

    def initialize(self, state_shape: Optional[int] = None, action_shape: Optional[int] = None, training: bool = True):
        self.state_shape = state_shape
        self.action_shape = action_shape
        if isinstance(self.model, str):
            self._build_registered_model(model_name=self.model, state_shape=state_shape, action_shape=action_shape)
        if training:
            self.configure_optimizer(self.optimizer, lr=self.lr)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    def play(self, env, num_eps=1, save_video=True, save_dir=None):
        self.model.eval()
        if isinstance(env, gym.Env):
            if save_video:
                env = RecordVideo(env, video_folder=save_dir, name_prefix="video")
            env = register_env(env)

        if isinstance(env, EnvWrapper):
            for _ in range(num_eps):
                states = env.reset()
                terminated = False
                ep_reward = 0
                while not terminated:
                    actions = self.act(states[0])
                    states, reward, terminated, _, _ = env.step(actions)
                    ep_reward += reward
        else:
            notice.warning("'env' must be an instance of 'EnvWrapper'")

    @staticmethod
    def save(ckpt, path):
        torch.save(ckpt, path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def make_ckpt(self, timestep, total_reward):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr": self.lr,
            "timestep": timestep,
            "total_reward": total_reward,
        }

    def on_timestep_start(self):
        pass

    def on_timestep_end(self):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self):
        pass

    def on_train_start(self):
        self.model.train()

    def on_train_end(self):
        pass

    def configure_optimizer(self, optimizer: Union[str, torch.optim.Optimizer], lr: Optional[float] = None):
        lr = get_param(lr, "lr", self.cfg, show=is_debug(self.cfg))
        self.optimizer = get_optimizer(optimizer, self.cfg)(self.model.parameters(), lr=lr)

    def _build_registered_model(self, model_name, state_shape, action_shape):
        self.model = registered_models[model_name](state_shape, action_shape)
