import torch
from room.train.agents.agent import OnPolicyAgent


class A2C(OnPolicyAgent):
    def __init__(self, policy, cfg, *args, **kwargs):
        super().__init__(policy, cfg, *args, **kwargs)

    def act(self, obss: torch.Tensor):
        with torch.no_grad():
            actions, values, infos = self.policy(obss)

    def update(self):
        pass
