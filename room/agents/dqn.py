import copy
from typing import Any, Dict, Optional, Union

import ray
import torch
from torch import nn

from room.agents import Agent
from room.common.aliases import registered_criteria
from room.common.typing import CfgType
from room.common.utils import get_param, is_debug
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger


@ray.remote
class DQN(Agent):
    def __init__(
        self,
        env: Union[str, EnvWrapper],
        model: Union[Any, str],
        optimizer: Union[torch.optim.Optimizer, str] = None,
        device: Optional[Union[str, torch.device]] = None,
        id: int = 0,
        cfg: Optional[CfgType] = None,
        lr: Optional[float] = None,
        criterion: Optional[Union[str, nn.Module]] = nn.HuberLoss,
        epsilon: Optional[float] = None,
        gamma: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            env=env,
            model=model,
            optimizer=optimizer,
            device=device,
            id=id,
            cfg=cfg,
            lr=lr,
            *args,
            **kwargs,
        )

        self.criterion = get_param(
            criterion, "criterion", cfg, registered_criteria, show=is_debug(cfg)
        )()
        self.epsilon = get_param(epsilon, "epsilon", cfg, show=is_debug(cfg))
        self.gamma = get_param(gamma, "gamma", cfg, show=is_debug(cfg))
        self.q_net = self.model
        self.target_q_net = copy.deepcopy(self.q_net)

    def act(self, state: torch.Tensor, step: int = 0) -> torch.Tensor:
        with torch.no_grad():
            q = self.q_net(state)
            action = torch.argmax(q)

        # eposilon greedy
        if self.model.training:
            threshold = self.epsilon * (1 - step / self.cfg["timesteps"])  # Linear decay
            if torch.rand(1) < threshold:
                action = torch.randint(0, self.action_shape, (1,))

        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = self.q_net(batch["states"])
        q = q.gather(1, batch["actions"]).squeeze(1)  # Select q value for action taken
        q_next = self.target_q_net(batch["next_states"]).max(1)[0]
        q_target = batch["rewards"].squeeze(1) + self.gamma * q_next
        loss = self.criterion(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def make_ckpt(self, timestep, total_reward):
        return {
            "model": self.q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr": self.lr,
            "timestep": timestep,
            "total_reward": total_reward,
        }
