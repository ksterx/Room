from abc import ABC, abstractmethod

import torch
from torch import nn

from .blocks import FCBN


class Policy(nn.Module, ABC):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def predict(self, obs):
        pass


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


def ActorCriticPolicy(Policy):
    def __init__(self, obs_space, action_space, hidden_size, share_backbone=True):
        super.__init__()
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.share_backbone = share_backbone

        if share_backbone:
            self.backbone = FCBN(layers=[obs_space.shape[0], hidden_size, hidden_size])
            self.actor_head = nn.Linear(self.hidden_size, self.action_space.n)
            self.critic_head = nn.Linear(self.hidden_size, 1)
        else:
            self.actor_net = FCBN(layers=[obs_space.shape[0], hidden_size, hidden_size, action_space.n])
            self.critic_net = FCBN(layers=[obs_space.shape[0], hidden_size, hidden_size, 1])

    def forward(self, obs):
        if self.share_backbone:
            x = self.backbone(obs)
            action_probs = self.actor_head(x)
            value = self.critic_head(x)
        else:
            action_probs = self.actor_net(obs)
            value = self.critic_net(obs)
        return action_probs, value

    @abstractmethod
    def predict(self, obs):
        pass

    def evaluate_actions(self, obs, action):
        """Evaluate actions according to the current policy,
        given the observations.

        Args:
            obs (_type_): _description_
            actions (_type_): _description_
        """
        pass


class VisionActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, obs_space, action_space, share_backbone=True):
        super.__init__(obs_space, action_space, share_backbone)

    def forward(self, obs):
        pass

    def predict(self, obs):
        pass

    def evaluate_actions(self, obs, action):
        pass
