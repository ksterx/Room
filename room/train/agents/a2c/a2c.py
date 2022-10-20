from typing import Union

import torch
from omegaconf import DictConfig
from room.train.agents.agent import OnPolicyAgent, wrap_param
from room.train.policies.policy import Policy


class A2C(OnPolicyAgent):
    agent_name = "a2c"

    def __init__(
        self,
        policy: Union[Policy, str],
        cfg: DictConfig,
        gamma: float = None,
        gae_lambda: float = None,
        entropy_coef: float = None,
        value_loss_coef: float = None,
        max_grad_norm: float = None,
        use_sde: bool = None,
        sde_sample_freq: int = None,
        normalize_advantage: bool = None,
        *args,
        **kwargs,
    ):

        gamma, cfg = wrap_param(cfg, gamma, "gamma")
        gae_lambda, cfg = wrap_param(cfg, gae_lambda, "gae_lambda")
        entropy_coef, cfg = wrap_param(cfg, entropy_coef, "entropy_coef")
        value_loss_coef, cfg = wrap_param(cfg, value_loss_coef, "value_loss_coef")
        max_grad_norm, cfg = wrap_param(cfg, max_grad_norm, "max_grad_norm")
        use_sde, cfg = wrap_param(cfg, use_sde, "use_sde")
        sde_sample_freq, cfg = wrap_param(cfg, sde_sample_freq, "sde_sample_freq")
        self.normalize_advantage, cfg = wrap_param(cfg, normalize_advantage, "normalize_advantage")

        super().__init__(
            policy=policy,
            cfg=cfg,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            *args,
            **kwargs,
        )

    def act(self, obss: torch.Tensor):
        with torch.no_grad():
            actions, values, infos = self.policy(obss)

    def update(self):
        pass
