from abc import ABC, abstractmethod

import torch
from torch import nn

from .blocks import FCBN


class Policy(nn.Module, ABC):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward(self, observation):
        pass

    @abstractmethod
    def predict(self, observation):
        pass


def ActorCriticPolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size, share_backbone=True):
        super.__init__()
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.share_backbone = share_backbone

        if share_backbone:
            self.backbone = FCBN(layers=[observation_space.shape[0], hidden_size, hidden_size])
            self.actor_head = nn.Linear(self.hidden_size, self.action_space.n)
            self.critic_head = nn.Linear(self.hidden_size, 1)
        else:
            self.actor_net = FCBN(layers=[observation_space.shape[0], hidden_size, hidden_size, action_space.n])
            self.critic_net = FCBN(layers=[observation_space.shape[0], hidden_size, hidden_size, 1])

    def forward(self, observation):
        if self.share_backbone:
            x = self.backbone(observation)
            action_probs = self.actor_head(x)
            value = self.critic_head(x)
        else:
            action_probs = self.actor_net(observation)
            value = self.critic_net(observation)
        return action_probs, value

    @abstractmethod
    def predict(self, observation):
        pass

    def evaluate_actions(self, observation, action):
        """Evaluate actions according to the current policy,
        given the observations.

        Args:
            observations (_type_): _description_
            actions (_type_): _description_
        """
        values,


class VisionActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, share_backbone=True):
        super.__init__(observation_space, action_space, share_backbone)

    def forward(self, observation):
        pass

    def predict(self, observation):
        pass

    def evaluate_actions(self, observation, action):
        pass
