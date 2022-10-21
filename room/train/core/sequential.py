from typing import Union

import torch
from room.train.core.trainer import Trainer
from tqdm import trange


class SequentialTrainer(Trainer):
    def __init__(self, env, agents: Union[tuple, list], cfg, callbacks=None):
        super().__init__(env, agents, cfg, callbacks)

    def train(self):
        super.train()

        states = self.env.reset()

        for t in trange(self.cfg.max_timesteps):

            for agent in self.agents:
                agent.on_before_step(timestep=t)

            # Get action tensors and stack them
            with torch.no_grad():
                actions = torch.vstack([agent.act(state) for agent, state in zip(self.agents, states)])
            next_states, rewards, dones, info = self.env.step(actions)

            with torch.no_grad():
                for agent in self.agents:
                    agent.collect(new_states, action, reward)

    def eval(self):
        super.eval()

    def save(self):
        pass
