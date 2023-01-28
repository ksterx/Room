import torch

from room.agents.policies.base import Policy
from room.agents.policies.blocks import FCBN


class EpsilonGreedyPolicy(Policy):
    def __init__(self, obs_space, action_space, hidden_size, epsilon=0.1):
        super.__init__()
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.net = FCBN(layers=[obs_space.shape[0], hidden_size, hidden_size, action_space.n])

    def forward(self, obs):
        return self.net(obs)

    def predict(self, obs):
        if torch.rand(1) < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                action_probs = self.net(obs)
                return torch.argmax(action_probs).item()

    def evaluate_actions(self, obs, action):
        """Evaluate actions using the policy

        Args:
            obs (torch.Tensor): The observation to evaluate actions for
            action (torch.Tensor): The action to evaluate

        Returns:
            torch.Tensor: The log probability of the action
            torch.Tensor: The entropy of the action
        """
        action_probs = self.net(obs)
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        entropy = -(action_probs * torch.log(action_probs)).sum(-1)
        return log_prob, entropy

    def get_action
