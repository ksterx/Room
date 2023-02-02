from typing import Any, Dict, List, Optional, Union

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
    show: Optional[bool] = True,
):
    show = False if show is None else show

    # If the value is None, load the value from the config file.
    if value is None:
        try:
            value = cfg[param_name]
            if aliases is not None:
                value = aliases[value]
        except KeyError:
            raise KeyError(f"{param_name} is not defined in the config file.")
        except TypeError:
            raise TypeError(f"{param_name}: None. No config file is provided.")

    # If the value is a string, load the value from the aliases.
    elif isinstance(value, str):
        if aliases is None:
            try:
                value = cfg[param_name]
                notice.debug(f"{param_name}: {value} is loaded from the config file.", show=show)
            except KeyError:
                raise KeyError(f"No {param_name} is not defined in the config file.")
            except TypeError:
                notice.warning(
                    f"{param_name}: {value}. Neither config file nor aliases are provided."
                )
        else:
            try:
                value = aliases[value]
                notice.debug(
                    f"{param_name}: {value} is loaded from the aliases: {aliases}", show=show
                )
            except KeyError:
                raise KeyError(f"No {param_name} is not defined in the aliases.")

    # If the value is not None or a string, return the value.
    else:
        pass

    return value


def get_device(device: Optional[Union[str, int, torch.device]]):
    if isinstance(device, torch.device):
        pass
    else:
        gpus = torch.cuda.device_count()

        if isinstance(device, str):
            if device == "cpu":
                device = torch.device("cpu")
                notice.info(f"Using CPU. {gpus} GPUs are available.")
            elif device == "cuda" and gpus == 0:
                device = torch.device("cpu")
                notice.warning("Using CPU. No GPU is available.")
            elif device == "cuda" and gpus > 0:
                device = torch.device(device)
                notice.info(f"Using GPU: {torch.cuda.current_device()}.")
            else:
                raise ValueError(
                    f"Device {device} is not found. Available devices are 'cpu' and 'cuda'"
                )

        elif isinstance(device, int):
            if device >= gpus:
                device = torch.device("cpu")
                notice.warning(f"Using CPU. GPU: {device} is not available.")
            else:
                device = torch.device(f"cuda:{device}")
                notice.info(f"Using GPU: {device}.")

        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        else:
            raise TypeError(f"Device type {type(device)} is not supported")

    return device


def is_debug(cfg):
    if cfg is None:
        return False
    else:
        return cfg["debug"]


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
