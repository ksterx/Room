from collections import OrderedDict
from typing import List, NamedTuple

from torch import nn


class InputOutputSize(NamedTuple):
    input_size: int
    output_size: int


def activation_from_str(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Activation {activation} not supported")


class FC(nn.Module):
    def __init__(self, layers: List[int], activation: str, softmax: bool, *args, **kwargs):
        """Fully connected block with batch normalization

        Args:
            layers (List[int]): Description of size. [input, hidden, hidden, ..., output]
            activation (str, optional): Activation function
            softmax (bool, optional): Whether to add a softmax layer at the end

        Example:
            >>> net = FCBN(layers=[10, 100, 100, 2])
        """

        super.__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)):
            if i < len(layers) - 1:
                self.net.add_module(f"fc{i+1}", nn.Linear(layers[i], layers[i + 1]))
                self.net.add_module(f"{activation}{i+1}", activation_from_str(activation))
            elif i == len(layers) - 1 and softmax:
                self.net.add_module(f"softmax{i+1}", nn.Softmax(dim=1))
            else:
                raise ValueError("Unexpected error. e.g. Layers list must be of length 3 or more")

    def forward(self, x):
        return self.net(x)


class FCBN(nn.Module):
    def __init__(self, layers: List[int], activation: str, softmax: bool, *args, **kwargs):
        """Fully connected block with batch normalization

        Args:
            layers (List[int]): Description of size. [input, hidden, hidden, ..., output]
            activation (str, optional): Activation function
            softmax (bool, optional): Whether to add a softmax layer at the end

        Example:
            >>> net = FCBN(layers=[10, 100, 100, 2])
        """

        super.__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)):
            if i < len(layers) - 1:
                self.net.add_module(f"fc{i+1}", nn.Linear(layers[i], layers[i + 1]))
                self.net.add_module(f"bn{i+1}", nn.BatchNorm1d(layers[i + 1]))
                self.net.add_module(f"{activation}{i+1}", activation_from_str(activation))
            elif i == len(layers) - 1 and softmax:
                self.net.add_module(f"softmax{i+1}", nn.Softmax(dim=1))
            else:
                raise ValueError("Unexpected error. e.g. Layers list must be of length 3 or more")

    def forward(self, x):
        return self.net(x)
