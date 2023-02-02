from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import gym
import ray
import torch
from gym.wrappers import RecordVideo
from omegaconf import DictConfig
from torch import nn, optim

from room import notice
from room.agents.policies import Policy, policies
from room.common import registered_criteria, registered_optimizers
from room.common.callbacks import Callback
from room.common.utils import get_device, get_param, is_debug
from room.envs import build_env
from room.envs.utils import get_action_shape, get_obs_shape
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.networks import registered_models


class Agent(ABC):
    def __init__(
        self,
        env: Union[str, EnvWrapper],
        model: Optional[Union[Policy, str]] = None,
        lr: Optional[float] = None,
        optimizer: Union[torch.optim.Optimizer, str] = None,
        criterion: Optional[Union[str, nn.Module]] = None,
        device: Optional[Union[str, torch.device]] = None,
        id: int = 0,
        cfg: Optional[Dict] = None,
        logger: Optional[Logger] = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
        *args,
        **kwargs,
    ):
        self.env = build_env(env)
        self.obs_shape = get_obs_shape(self.env.observation_space)
        self.action_shape = get_action_shape(self.env.action_space)
        self.model = get_param(model, "model", cfg, show=is_debug(cfg))
        if isinstance(self.model, str):
            self.model = registered_models[self.model](self.obs_shape, self.action_shape)
        self.lr = get_param(lr, "lr", cfg, show=is_debug(cfg))
        self.optimizer = get_param(optimizer, "optimizer", cfg, show=is_debug(cfg))
        if isinstance(self.optimizer, str):
            self.optimizer = registered_optimizers[self.optimizer](
                self.model.parameters(), lr=self.lr
            )
        self.device = get_device(device)
        self.id = id
        self.cfg = cfg
        self.logger = logger
        self.criterion = get_param(
            criterion, "criterion", cfg, registered_criteria, show=is_debug(cfg)
        )
        self.callbacks = callbacks
        self.total_reward = 0.0
        self.episode = 0

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def step(self):
        pass

    def reset_env(self):
        self.total_reward = 0.0
        state, info = self.env.reset()
        return state, info

    def get_next_state(self, next_state, terminated, truncated):
        if terminated.any() or truncated.any():
            next_state, _ = self.reset_env()
        return next_state

    def play(self, env, num_eps=1, save_video=True, save_dir=None):
        self.model.eval()
        if isinstance(env, gym.Env):
            if save_video:
                env = RecordVideo(env, video_folder=save_dir, name_prefix="video")
            env = build_env(env)

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

    def get_model(self):
        return self.model

    def set_logger(self):
        pass


@ray.remote
class Actor(ABC):
    def __init__(self, id, model, env, cfg, logger, callbacks, *args, **kwargs):
        super().__init__()

        self.id = id
        self.env = build_env(env)
        self.obs_shape = get_obs_shape(self.env.observation_space)
        self.action_shape = get_action_shape(self.env.action_space)
        self.model = get_param(model, "model", cfg, show=is_debug(cfg))
        if isinstance(self.model, str):
            self.model = registered_models[self.model](self.obs_shape, self.action_shape)
        self.model.eval()
        self.args = args
        self.kwargs = kwargs

    def explore(self, weights):
        self.model.load_state_dict(weights)
        states = self.env.reset()

    def act(self, state):
        pass

    def play(self):
        pass


class Learner(ABC):
    def __init__(
        self, id, optimizer, lr, criterion, cfg, device, logger, callbacks, *args, **kwargs
    ):
        super().__init__()

        self.id = id
        self.lr = get_param(lr, "lr", cfg, show=is_debug(cfg))
        self.cfg = cfg
        self.lr = get_param(lr, "lr", cfg, show=is_debug(cfg))
        self.optimizer = get_param(optimizer, "optimizer", cfg, show=is_debug(cfg))
        if isinstance(self.optimizer, str):
            self.optimizer = registered_optimizers[self.optimizer](
                self.model.parameters(), lr=self.lr
            )
        self.criterion = get_param(
            criterion, "criterion", cfg, registered_criteria, show=is_debug(cfg)
        )
