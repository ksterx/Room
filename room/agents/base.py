from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import gym
import ray
import torch
from gym.wrappers import RecordVideo
from omegaconf import DictConfig
from torch import optim

from room import notice
from room.agents.policies import Policy, policies
from room.common import registered_criteria, registered_optimizers
from room.common.callbacks import Callback
from room.common.utils import get_device, get_param, is_debug
from room.envs import build_env
from room.envs.utils import get_action_shape, get_obs_shape
from room.envs.wrappers import EnvWrapper
from room.networks import registered_models


class Agent(ABC):
    def __init__(
        self,
        env: Union[str, EnvWrapper],
        model: Optional[Union[Policy, str]] = None,
        optimizer: Union[torch.optim.Optimizer, str] = None,
        device: Optional[Union[str, torch.device]] = None,
        id: int = 0,
        cfg: Optional[Dict] = None,
        lr: Optional[float] = None,
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
        self.optimizer = get_param(
            optimizer, "optimizer", cfg, registered_optimizers, show=is_debug(cfg)
        )
        self.device = get_device(device)
        self.id = id
        self.cfg = cfg
        self.lr = lr

        self.callbacks = callbacks

        try:
            for callback in self.callbacks:
                callback.on_agent_init(agent_id=self.id)
        except TypeError:
            self.callbacks.on_agent_init(*args, **kwargs)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    def reset_env(self):
        self.env.reset()

    def rollout(self):
        pass

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

    def configure_optimizer(
        self, optimizer: Union[str, torch.optim.Optimizer], lr: Optional[float] = None
    ):
        lr = get_param(lr, "lr", self.cfg, show=is_debug(self.cfg))
        self.optimizer = self.get_optimizer(optimizer, self.cfg)(self.model.parameters(), lr=lr)

    def _build_registered_model(self, model_name, state_shape, action_shape):
        self.model = registered_models[model_name](state_shape, action_shape)

    def get_model(self):
        return self.model
