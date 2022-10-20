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
