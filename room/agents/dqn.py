from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from room.agents.base import Agent
from room.agents.policies.base import Policy
from room.common.typing import CfgType
from room.common.utils import get_optimizer, get_param
from room.loggers import Logger
from room.memories.base import Memory
from room.networks.blocks import MLP


class DQN(Agent):
    def __init__(
        self,
        model: Union[Any, str],
        optimizer: Union[torch.optim.Optimizer, str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: CfgType = None,
        logger: Optional[Logger] = None,
        lr: Optional[float] = None,
        epsilon: Optional[float] = None,
        gamma: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            logger=logger,
        )

        self.epsilon = get_param(epsilon, "epsilon", cfg)
        self.gamma = get_param(gamma, "gamma", cfg)
        self.optimizer = optimizer
        self.lr = lr

        self.loss_fn = F.smooth_l1_loss  # TODO: get from cfg or parse from name

    def act(self, state: torch.Tensor) -> torch.Tensor:
        q = self.target_q_net(state)
        action = torch.argmax(q)

        # eposilon greedy
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, self.action_dim, (1,))  # TODO: Fix this
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.model.train()
        q = self.q_net(batch["state"])
        q = q.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        q_next = self.target_q_net(batch["next_state"])
        q_next = q_next.max(1)[0]
        q_target = batch["reward"].squeeze(1) + self.gamma * q_next
        loss = self.loss_fn(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def on_before_train(self, state_dim: Optional[int] = None, action_dim: Optional[int] = None):
        super().on_before_train(state_dim, action_dim)
        self.q_net = self.model
        self.target_q_net = self.model
        self.configure_optimizer(self.optimizer, self.lr)
