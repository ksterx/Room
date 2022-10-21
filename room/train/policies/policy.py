from abc import ABC, abstractmethod

import torch.nn as nn


class Policy(nn.Module, ABC):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward(self, obss):
        pass

    @abstractmethod
    def predict(self, obss):
        pass


def ActorCritic(Policy):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def forward(self, obs):
        pass

    @abstractmethod
    def predict(self, obs):
        pass

    def evaluate_actions(self, observations, actions):
        """Evaluate actions according to the current policy,
        given the observations.

        Args:
            observations (_type_): _description_
            actions (_type_): _description_
        """
        values,
