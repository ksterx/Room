from typing import Optional, Union

import torch

from room.agents.base import Agent
from room.agents.policies.base import Policy
from room.common.typing import CfgType, Dict
from room.loggers import Logger
from room.memories.base import Memory


class DQN(Agent):
    def __init__(
        self,
        policy: Union[Policy, str],
        cfg: CfgType,
        logger: Optional[Logger] = None,
        *args,
        **kwargs,
    ):
        super().__init__(policy, cfg, logger)
        self.q_net = self.policy
        self.target_q_net = self.q_net.clone()
        self.loss = None  # TODO: add loss

    def act(self, state: torch.Tensor) -> torch.Tensor:
        q = self.target_q_net(state)
        action = torch.argmax(q)

        # eposilon greedy
        if torch.rand(1) < self.cfg.agent.epsilon:
            action = torch.randint(0, self.cfg.agent.action_dim, (1,))
        return action

    def learn(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = self.q_net(batch["state"])
        q = q.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
        q_next = self.target_q_net(batch["next_state"])
        q_next = q_next.max(1)[0]
        q_target = batch["reward"] + self.cfg.agent.gamma * q_next
        loss = self.loss(q, q_target)
