from typing import Any, Dict, Optional, Tuple, Union

import gym
import torch
from gym import spaces

from room import notice


def get_obs_shape(obs_space: Union[int, Tuple[int], gym.Space]) -> int:
    """Get the shape of the observation

    Refered from stable-baselines3/common/preprocessing.py
    (https://github.com/DLR-RM/stable-baselines3/blob/8452106734ba1749cc4ddd5ae9fe7fd28ca55bf7/stable_baselines3/common/preprocessing.py)

    Args:
        obs_space (Union[int, Tuple[int], gym.Space])

    Returns:
        Shape of the observation (int)
    """
    if isinstance(obs_space, int):
        return obs_space
    elif isinstance(obs_space, tuple):
        raise NotImplementedError(f"Space type {type(obs_space)} is not supported")
    elif isinstance(obs_space, spaces.Box):
        if len(obs_space.shape) == 1:
            return obs_space.shape[0]
        else:
            raise NotImplementedError(f"Space type {type(obs_space)} is not supported")


def get_action_shape(action_space: Union[int, Tuple[int], gym.Space]) -> int:
    """Get the dimension of the action space.

    Refered from stable-baselines3/common/preprocessing.py
    (https://github.com/DLR-RM/stable-baselines3/blob/8452106734ba1749cc4ddd5ae9fe7fd28ca55bf7/stable_baselines3/common/preprocessing.py)

    Args:
        action_space (Union[int, Tuple[int], gym.Space])

    Returns:
        Dimension of the action space (int)
    """
    if isinstance(action_space, int):
        return action_space
    elif isinstance(action_space, tuple):
        raise NotImplementedError(f"Space type {type(action_space)} is not supported")
    elif isinstance(action_space, spaces.Box):
        raise NotImplementedError(f"Space type {type(action_space)} is not supported")
    elif isinstance(action_space, spaces.Discrete):
        return action_space.n
    else:
        raise NotImplementedError(f"Space type {type(action_space)} is not supported")
