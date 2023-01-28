from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import gym
from omegaconf import DictConfig

from room.memories.base import Memory
from room.policies import Policy, policies


def wrap_param(cfg: DictConfig, param, param_name: str):
    # Read param from config
    if param is None:
        param = cfg.agent[param_name]
    # Override with an function argument
    else:
        cfg.agent[param_name] = param

    return param, cfg


class Agent(ABC):
    def __init__(self, policy: Union[str, Policy], cfg: Optional[Dict] = None, *args, **kwargs):
        self.cfg = cfg

        if isinstance(policy, str):
            self.policy = policies[policy]
        elif isinstance(policy, Policy):
            self.policy = policy
        else:
            raise TypeError("policy must be a string or a Policy object")

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


class OnPolicyAgent(Agent):
    def __init__(
        self,
        env,
        policy,
        cfg: DictConfig,
        gamma: float,
        gae_lambda: float,
        entropy_coef: float,
        value_loss_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        num_steps: int,
        *args,
        **kwargs,
    ):

        super().__init__(
            env=env,
            policy=policy,
            cfg=cfg,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            *args,
            **kwargs,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.num_steps = num_steps

        self.memory = RolloutMemory(
            memory_size=num_steps, num_envs=self.num_envs, obs_space=self.obs_space, action_space=self.action_space
        )

    @abstractmethod
    def act(self, obss):
        pass

    def collect(self, obs, action, reward, done, log_prob):
        self.memory.add(obs, action, reward, done, log_prob)

    @abstractmethod
    def update(self):
        pass


class OffPolicyAgent(Agent):
    def __init__(self, obs_space, act_space, cfg):
        super().__init__(obs_space, act_space, cfg)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def collect(self):
        pass

    @abstractmethod
    def update(self):
        pass
