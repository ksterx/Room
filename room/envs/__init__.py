from typing import Union

import gym

from room import notice
from room.envs.loaders import (
    load_isaacgym_env_preview2,
    load_isaacgym_env_preview3,
    load_isaacgym_env_preview4,
    load_omniverse_isaacgym_env,
)
from room.envs.wrappers import EnvWrapper, GymEnvWrapper, IsaacGymPreview3EnvWrapper


def build_env(env: Union[str, gym.Env, gym.Wrapper], verbose=True, **kwargs) -> EnvWrapper:

    if isinstance(env, str):
        try:
            render = kwargs["render"]
        except KeyError:
            render = False
        if render:
            render_mode = "human"
        elif not render:
            render_mode = None
        else:
            raise ValueError("'render' must be bool")

        if env in registered_envs:
            if registered_envs[env]["type"] == "gym":
                notice.info("Environment type: Gym") if verbose else None
                return GymEnvWrapper(gym.make(registered_envs[env]["env_name"], render_mode=render_mode))
            elif registered_envs[env]["type"] == "omniverse":
                notice.info("Environment type: OmniIsaacGym") if verbose else None
                raise NotImplementedError("OmniIsaacGym not implemented")
        else:
            raise ValueError(f"Environment {env} not registered")

    elif isinstance(env, gym.Env) or isinstance(env, gym.Wrapper):
        notice.info("Environment type: Gym") if verbose else None
        return GymEnvWrapper(env)
    # TODO: DeepMind Environment
    # TODO: OmniIsaacGym Environment
    else:
        try:
            notice.info("Environment type: IsaacGym") if verbose else None
            return IsaacGymPreview3EnvWrapper(env)
        except TypeError:
            notice.error("Environment type not supported")
            quit()


registered_envs = {
    "cartpole": {"type": "gym", "env_name": "CartPole-v1"},
    "Pendulum": {"type": "gym", "env_name": "Pendulum-v1"},
}
