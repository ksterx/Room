from typing import Optional, Union

import torch
import torch.nn.functional as F

from room.agents.base import Agent
from room.agents.policies.base import Policy
from room.common.typing import CfgType, Dict
from room.common.utils import get_optimizer, get_param
from room.loggers import Logger
from room.memories.base import Memory
from room.networks.blocks import MLP


class DQN(Agent):
    def __init__(
        self,
        policy: Union[Policy, str],
        optimizer: Union[torch.optim.Optimizer, str],
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
            policy=policy,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            logger=logger,
        )

        self.epsilon = get_param(epsilon, "epsilon", cfg)
        self.gamma = get_param(gamma, "gamma", cfg)

        self.policy = MLP([4, 100, 2], activation="relu")  # TODO: Fix this
        self.configure_optimizer(optimizer, lr)
        self.q_net = self.policy
        self.target_q_net = self.policy
        self.loss_fn = F.smooth_l1_loss  # TODO: get from cfg or parse from name

    def act(self, state: torch.Tensor) -> torch.Tensor:
        q = self.target_q_net(state)
        action = torch.argmax(q)

        # eposilon greedy
        if torch.rand(1) < self.epsilon:
            action = torch.randint(0, 2, (1,))  # TODO: Fix this
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = self.q_net(batch["state"])
        q = q.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        q_next = self.target_q_net(batch["next_state"])
        q_next = q_next.max(1)[0]
        q_target = batch["reward"] + self.gamma * q_next
        loss = self.loss_fn(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
