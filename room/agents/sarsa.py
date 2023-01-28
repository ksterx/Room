from room.agents.base import OnPolicyAgent, wrap_param
from room.memories.base import Memory
from room.policies.base import Policy


class SARSA(OnPolicyAgent):
    def __init__(self, policy: Policy, memory: Memory, gamma: float = 0.99, tau: float = 1.0):
        super().__init__(policy, memory, gamma, tau)

    def act(self, observations):
        return self.policy.act(observations)

    def learn(self):
        pass
