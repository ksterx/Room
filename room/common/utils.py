from typing import Any, Dict, Optional, Union

import torch
from torch import optim

from room import notice

# from room.agents.base import Agent
# from room.agents.policies import registered_policies

# def check_agent(agent: Agent):
#     if isinstance(agent.policy_name, str):

#         if agent.policy not in registered_policies[agent.name]:
#             raise ValueError(
#                 f"'{agent.policy_name}' is not available for {agent.name}. Available policies are {registered_policies[agent.name]}"
#             )


def get_param(
    value: Any,
    param_name: Optional[str],
    cfg: Optional[Dict[str, Any]],
    aliases: Optional[Dict[str, Any]] = None,
    verbose: Optional[bool] = True,
):
    verbose = False if verbose is None else verbose

    if value is None:
        try:
            value = cfg[param_name]
            notice.debug(f"Loading default {param_name}. It is defined in the config file.", render=verbose)
        except KeyError:
            raise KeyError(f"{param_name} is not defined in the config file.")
        except TypeError:
            notice.warning(f"{param_name}: None. No config file is provided.")

    elif isinstance(value, str):
        if aliases is None:
            try:
                value = cfg[param_name]
                notice.debug(f"{param_name}: {value} is loaded from the config file.", render=verbose)
            except KeyError:
                raise KeyError(f"No {param_name} is not defined in the config file.")
            except TypeError:
                notice.warning(f"{param_name}: {value}. Neither config file nor aliases are provided.")
        else:
            notice.debug(f"{param_name}: {value} is loaded from the aliases", render=verbose)
            try:
                value = aliases[value]
            except KeyError:
                raise KeyError(f"No {param_name} is not defined in the aliases.")

    else:
        if aliases is None:
            try:
                if value == cfg[param_name]:
                    notice.debug(f"Loading {param_name} same as the config file.", render=verbose)
                else:
                    notice.debug(
                        f"Loading {param_name} from the function argument instead of the config file.", render=verbose
                    )
            except KeyError:
                notice.warning(f"No {param_name} is not defined in the config file.")
        else:
            if value in aliases.values():
                notice.debug(f"Loading {param_name} same as the aliases.", render=verbose)
            else:
                notice.warning(f"No {param_name} is not defined in the aliases.")

    return value


def get_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, torch.device):
        pass
    else:
        raise TypeError(f"Device type {type(device)} is not supported")

    return device


def get_optimizer(optimizer: Optional[Union[str, optim.Optimizer]], cfg: Optional[dict] = None):
    optim_aliases = {
        "adam": optim.Adam,
        "rmsprop": optim.RMSprop,
        "sgd": optim.SGD,
        "adagrad": optim.Adagrad,
    }
    optimizer = get_param(optimizer, "optimizer", cfg, optim_aliases)
    return optimizer


def flatten_dict(d: dict):
    """Flatten a nested dictionary into a single dictionary."""
    out = {}

    def flatten(item, name=""):
        if type(item) is not dict:
            try:
                del out["name"]
            except KeyError:
                pass
            key = name[:-1]
            if key in out.keys():
                raise ValueError(f"Key {key} is duplicated.")
            else:
                out[name[:-1]] = item
        else:
            for a in item:
                flatten(item[a], a + ".")

    flatten(d)
    return out
