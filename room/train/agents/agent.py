from abc import ABC, abstractmethod

from omegaconf import DictConfig
from room.train.agents.memory import Memory, RolloutMemory


def wrap_param(cfg: DictConfig, param, param_name: str):
    # Read param from config
    if param is None:
        param = cfg.agent[param_name]
    # Override with an function argument
    else:
        cfg.agent[param_name] = param

    return param, cfg


class Agent(ABC):
    def __init__(self, policy, obs_space, act_space, cfg, *args, **kwargs):
        self.policy = policy
        self.obs_space = obs_space
        self.act_space = act_space
        self.cfg = cfg

    @abstractmethod
    def act(self, obss):
        pass


class OnPolicyAgent(Agent):
    def __init__(
        self,
        policy,
        cfg: DictConfig,
        gamma: float,
        gae_lambda: float,
        entropy_coef: float,
        value_loss_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        obs_space,
        act_space,
        *args,
        **kwargs,
    ):

        super().__init__(
            policy=policy,
            cfg=cfg,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            obs_space=obs_space,
            act_space=act_space,
            *args,
            **kwargs,
        )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.memory = RolloutMemory(cfg.memory_size, cfg.num_envs)

    @abstractmethod
    def act(self, obss):
        pass

    @abstractmethod
    def update(self):
        pass

    def log(self):
        pass


class OffPolicyAgent(Agent):
    def __init__(self, obs_space, act_space, cfg):
        super().__init__(obs_space, act_space, cfg)

    @abstractmethod
    def act(self):
        pass

    @abstractmethod
    def update(self):
        pass

    def log(self):
        pass
