import os
from typing import NamedTuple

import torch

__all__ = ["__version__", "notice"]

# read library version from file
_path = os.path.join(os.path.dirname(__file__), "version.txt")
with open(_path, "r") as file:
    __version__ = file.read().strip()


class Notice:
    _grey = "\x1b[37;40m"
    _blue = "\x1b[34;20m"
    _yellow = "\x1b[33;20m"
    _green = "\x1b[32;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"

    def debug(self, *msg, show=True):
        msg = " ".join(msg)
        if show:
            print(f"{self._blue}[DEBUG] {msg}{self._reset}")

    def info(self, *msg):
        msg = " ".join(msg)
        print(f"{self._grey}[INFO] {msg}{self._reset}")

    def warning(self, *msg):
        msg = " ".join(msg)
        print(f"{self._yellow}[WARNING] {msg}{self._reset}")

    def error(self, *msg):
        msg = " ".join(msg)
        print(f"{self._red}[ERROR] {msg}{self._reset}")

    def critical(self, *msg):
        msg = " ".join(msg)
        print(f"{self._bold_red}[CRITICAL] {msg}{self._reset}")


notice = Notice()


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: bool
