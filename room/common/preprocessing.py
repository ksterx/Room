from typing import Any, Dict, Optional, Tuple, Union

import gym
import torch

from room import notice


def get_space_shape(space: Union[int, Tuple[int], gym.Space]):
    if isinstance(space, int):
        return space
    elif isinstance(space, tuple):
        NotImplementedError(f"Space type {type(space)} is not supported")
    else:
        NotImplementedError(f"Space type {type(space)} is not supported")
