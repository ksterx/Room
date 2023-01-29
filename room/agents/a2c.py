from typing import Union

import torch
from omegaconf import DictConfig
from torch.nn import functional as F

from room import notice
from room.agents.base import Agent
from room.agents.policies import Policy
from room.common.utils import get_device, get_param
from room.memories.base import Memory


class A2C(Agent):
    name = "a2c"

    def __init__(
        self,
        env,
        policy: Union[Policy, str],
        cfg: DictConfig,
        gamma: float = None,
        gae_lambda: float = None,
        entropy_coef: float = None,
        value_loss_coef: float = None,
        max_grad_norm: float = None,
        use_sde: bool = None,
        sde_sample_freq: int = None,
        num_steps: int = None,
        normalize_advantage: bool = None,
        *args,
        **kwargs,
    ):

        # If paramters are not provided, read them from the config

        self.gamma = get_param(gamma, "gamma", cfg)
        self.gae_lambda = get_param(gae_lambda, "gae_lambda", cfg)
        self.entropy_coef = get_param(entropy_coef, "entropy_coef", cfg)
        self.value_loss_coef = get_param(value_loss_coef, "value_loss_coef", cfg)
        self.max_grad_norm = get_param(max_grad_norm, "max_grad_norm", cfg)
        self.use_sde = get_param(use_sde, "use_sde", cfg)
        self.sde_sample_freq = get_param(sde_sample_freq, "sde_sample_freq", cfg)
        self.num_steps = get_param(num_steps, "num_steps", cfg)
        self.normalize_advantage = get_param(normalize_advantage, "normalize_advantage", cfg)

        super().__init__(
            env=env,
            model=policy,
            cfg=cfg,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            num_steps=num_steps,
            *args,
            **kwargs,
        )

    def __str__(self):
        return f"A2C\n{self.model}"

    def act(self, states: torch.Tensor):
        with torch.no_grad():
            actions, values, infos = self.model(states)

    def update(self):
        rollouts = self.memory.sample()

        for rollout in rollouts:  # TODO

            values, log_probs, entropy = self.model.evaluate_actions(rollout.observations, rollout.actions)
            policy_loss = -(advantages * log_probs).mean()
            value_loss = F.mse_loss(returns, values)
            loss = policy_loss + self.entropy_coef * entropy_loss + self.value_loss_coef * value_loss

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # Avoid gradient explosion
            self.optimizer.step()

    # def step(self, states, actions, rewards, next_states, dones):
    #     self.memory.append(states, actions, rewards, next_states, dones)

    #     if self.memory.is_full():
    #         self.update()

    # def on_before_step(self, timestep):
    #     if timestep % self.sde_sample_freq == 0:
    #         self.policy.reset_noise()

    def save(self):
        pass

    def load(self):
        pass

    def eval(self):
        pass

    def train(self):
        pass
