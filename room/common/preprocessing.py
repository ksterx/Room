from typing import Any, Dict, Tuple, Union

import gym

from room import notice


def get_space_shape(space: Union[int, Tuple[int], gym.Space]):
    if isinstance(space, int):
        return space
    elif isinstance(space, tuple):
        NotImplementedError(f"Space type {type(space)} is not supported")
    else:
        NotImplementedError(f"Space type {type(space)} is not supported")


def get_param(item: Any, param_name: str, cfg: Dict, aliases: Dict = None):
    if item is None:
        notice.info(f"Loading default {param_name}. It is defined in the config file.")
        try:
            item = cfg[param_name]
        except KeyError:
            raise KeyError(f"{param_name} is not defined in the config file.")

    elif isinstance(item, str):
        if aliases is None:
            notice.info(f"Loading {param_name} from the config file.")
            try:
                item = cfg[param_name]
            except KeyError:
                raise KeyError(f"No {param_name} is not defined in the config file.")
        else:
            notice.info(f"Loading {param_name} from the aliases.")
            try:
                item = aliases[item]
            except KeyError:
                raise KeyError(f"No {param_name} is not defined in the aliases.")

    return item
