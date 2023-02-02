from abc import ABC, abstractmethod

import torch
from torch import nn

from room.networks.blocks import MLPBN


class Policy(nn.Module, ABC):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def predict(self, obs):
        pass


class ActorCriticPolicy(Policy):
    def __init__(self, obs_space, action_space, hidden_size, share_backbone=True):
        super.__init__()
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.share_backbone = share_backbone

        if share_backbone:
            self.backbone = MLPBN(layers=[obs_space.shape[0], hidden_size, hidden_size])
            self.actor_head = nn.Linear(self.hidden_size, self.action_space.n)
            self.critic_head = nn.Linear(self.hidden_size, 1)
        else:
            self.actor_net = MLPBN(
                layers=[obs_space.shape[0], hidden_size, hidden_size, action_space.n]
            )
            self.critic_net = MLPBN(layers=[obs_space.shape[0], hidden_size, hidden_size, 1])

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
