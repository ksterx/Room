from typing import Union

from .base import Trainer


class SequentialTrainer(Trainer):
    def __init__(self, env, agents: Union[tuple, list], cfg):
        super().__init__(env, agents, cfg)

    def train(self):
        pass

    def eval(self):
        pass

    def save(self):
        pass
