# MIT License
#
# Copyright (c) 2021 Toni-SM
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import gym
import numpy as np
import torch
from packaging import version
from room import logger


class EnvWrapper(ABC):
    def __init__(self, env: Any) -> None:
        self.env = env

        # Set compute device
        if hasattr(self.env, "device"):
            self.device = torch.device(self.env.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def reset(self) -> torch.Tensor:
        pass

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Step one time step in the environment

        Args:
            action (torch.Tensor): The action to take in the environment

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]: Tuple of (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    def num_envs(self) -> int:
        return self.env.num_envs if hasattr(self.env, "num_envs") else 1

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space


class IsaacGymEnvWrapper(EnvWrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)


class GymEnvWrapper(EnvWrapper):
    def __init__(self, env: Any) -> None:
        super().__init__(env)

        self.is_vectorized = False
        try:
            if isinstance(env, gym.vector.SyncVectorEnv) or isinstance(env, gym.vector.AsyncVectorEnv):
                self._vectorized = True
        except Exception as e:
            logger.warning(f"Failed to check for a vectorized environment: {e}")

        self.api_deprecated = version.parse(gym.__version__) < version.parse(" 0.25.0")
        if self.api_deprecated:
            logger.info(f"OpenAI Gym version: {gym.__version__} <= 0.25")
        else:
            raise NotImplementedError("OpenAI Gym's new API is not yet supported")
            # TODO: Implement new API

    def reset(self) -> torch.Tensor:
        NotImplemented  # TODO

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        if self.is_vectorized:
            obs, reward, done, info = self.env.step(actions)
        else:
            obs, reward, done, info = self.env.step(actions.item())
        return (
            torch.tensor(obs, device=self.device),
            torch.tensor(reward, device=self.device),
            torch.tensor(done, device=self.device),
            info,
        )

    def render(self) -> None:
        NotImplemented

    def close(self) -> None:
        NotImplemented

    @property
    def observation_space(self) -> gym.Space:
        if self.is_vectorized:
            return self.env.single_observation_space
        else:
            return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        if self.is_vectorized:
            return self.env.single_action_space
        else:
            return self.env.action_space

    def _convert_act_to_tensor(self, act: torch.Tensor):
        space = self.env.action_space if self.is_vectorized else self.action_space

        if self.is_vectorized:
            if isinstance(space, gym.spaces.MultiDiscrete):
                return np.array(act.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
            elif isinstance(space, gym.spaces.Tuple):
                if isinstance(space[0], gym.spaces.Box):
                    return np.array(act.cpu().numpy(), dtype=space[0].dtype).reshape(space.shape)
                elif isinstance(space[0], gym.spaces.Discrete):
                    return np.array(act.cpu().numpy(), dtype=space[0].dtype).reshape(-1)
        elif isinstance(space, gym.spaces.Discrete):
            return act.item()
        elif isinstance(space, gym.spaces.Box):
            return np.array(act.cpu().numpy(), dtype=space.dtype).reshape(space.shape)
        raise ValueError(f"Action space type {type(space)} not supported")

    def _convert_obs_to_tensor(self, observation: Any, space: Union[gym.Space, None] = None) -> torch.Tensor:
        observation_space = self._env.observation_space if self._vectorized else self.observation_space
        space = space if space is not None else observation_space

        if self._vectorized and isinstance(space, gym.spaces.MultiDiscrete):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, int):
            return torch.tensor(observation, device=self.device, dtype=torch.int64).view(self.num_envs, -1)
        elif isinstance(observation, np.ndarray):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Discrete):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Box):
            return torch.tensor(observation, device=self.device, dtype=torch.float32).view(self.num_envs, -1)
        elif isinstance(space, gym.spaces.Dict):
            ret = torch.cat(
                [self._observation_to_tensor(observation[k], space[k]) for k in sorted(space.keys())], dim=-1
            ).view(self.num_envs, -1)
            return ret
        else:
            raise ValueError("Observation space type {} not supported. Please report this issue".format(type(space)))


def register_env(env: Any, verbose=False) -> EnvWrapper:
    if isinstance(env, gym.core.Env) or isinstance(env, gym.core.Wrapper):
        logger.info("Environment type: Gym") if verbose else None
        return GymEnvWrapper(env)
    # TODO: DeepMind Environment
    # TODO: OmniIsaacGym Environment
    else:
        try:
            logger.info("Environment type: IsaacGym") if verbose else None
            return IsaacGymEnvWrapper(env)
        except TypeError:
            logger.error("Environment type not supported")
            quit()
