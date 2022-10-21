from typing import Union

import torch
from room import log
from room.train.core.trainer import Trainer
from tqdm import trange


class SequentialTrainer(Trainer):
    def __init__(self, env, agents: Union[tuple, list], cfg, logger, callbacks=None):
        super().__init__(env, agents, cfg, callbacks)

    def train(self, max_timesteps: int):
        super().train()

        states = self.env.reset()

        for t in trange(max_timesteps):

            for agent in self.agents:
                agent.on_before_step(timestep=t)

            # Get action tensor from each agent and stack them
            with torch.no_grad():
                log.debug(f"States: {states}")
                for agent, state in zip(self.agents, states):
                    log.debug(f"State: {state}, Agent: {agent}")
                actions = torch.vstack([agent.act(state) for agent, state in zip(self.agents, states)])
            next_states, rewards, dones, info = self.env.step(actions)

            with torch.no_grad():
                for agent in self.agents:
                    agent.collect(new_states, action, reward)

    def eval(self):
        super().eval()

    def save(self):
        pass
