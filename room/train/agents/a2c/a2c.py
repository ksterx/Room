from typing import Union

import torch
from omegaconf import DictConfig
from room.train.agents.agent import OnPolicyAgent, wrap_param
from room.train.agents.memory import RolloutMemory
from room.train.policies.policy import Policy
from torch.nn import functional as F


class A2C(OnPolicyAgent):
    agent_name = "a2c"

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
        gamma, cfg = wrap_param(cfg, gamma, "gamma")
        gae_lambda, cfg = wrap_param(cfg, gae_lambda, "gae_lambda")
        entropy_coef, cfg = wrap_param(cfg, entropy_coef, "entropy_coef")
        value_loss_coef, cfg = wrap_param(cfg, value_loss_coef, "value_loss_coef")
        max_grad_norm, cfg = wrap_param(cfg, max_grad_norm, "max_grad_norm")
        use_sde, cfg = wrap_param(cfg, use_sde, "use_sde")
        sde_sample_freq, cfg = wrap_param(cfg, sde_sample_freq, "sde_sample_freq")
        num_steps, cfg = wrap_param(cfg, num_steps, "num_steps")
        self.normalize_advantage, cfg = wrap_param(cfg, normalize_advantage, "normalize_advantage")

        super().__init__(
            env=env,
            policy=policy,
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

    def act(self, obss: torch.Tensor):
        with torch.no_grad():
            actions, values, infos = self.policy(obss)

    def update(self):
        rollouts = self.memory.get()

        for rollout in rollouts:  # TODO

            values, log_probs, entropy = self.policy.evaluate_actions(rollout.observations, rollout.actions)
            policy_loss = -(advantages * log_probs).mean()
            value_loss = F.mse_loss(returns, values)
            loss = policy_loss + self.entropy_coef * entropy_loss + self.value_loss_coef * value_loss

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # Avoid gradient explosion
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
