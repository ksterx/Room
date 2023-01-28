from typing import Any

from room import notice
from room.envs.loaders import (
    load_isaacgym_env_preview2,
    load_isaacgym_env_preview3,
    load_isaacgym_env_preview4,
    load_omniverse_isaacgym_env,
)
from room.envs.wrappers import EnvWrapper, GymEnvWrapper, IsaacGymEnvWrapper


def register_env(env: Any, verbose=False) -> EnvWrapper:
    if isinstance(env, gym.Env) or isinstance(env, gym.Wrapper):
        notice.info("Environment type: Gym") if verbose else None
        return GymEnvWrapper(env)
    # TODO: DeepMind Environment
    # TODO: OmniIsaacGym Environment
    else:
        try:
            notice.info("Environment type: IsaacGym") if verbose else None
            return IsaacGymEnvWrapper(env)
        except TypeError:
            notice.error("Environment type not supported")
            quit()
