from typing import List, Optional, Union

import torch
from tqdm import trange

from room import notice
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.utils import get_param
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory
from room.trainers.base import Trainer


class SimpleTrainer(Trainer):
    def __init__(
        self,
        env: EnvWrapper,
        agent: Agent,
        timesteps: Optional[int] = None,
        memory: Optional[Union[str, Memory]] = None,
        logger: Optional[Logger] = None,
        cfg: dict = None,
        callbacks: Union[Callback, List[Callback]] = None,
    ):
        super().__init__(env, agent, timesteps, memory, logger, cfg, callbacks)
        self.agent = agent

    def train(self):
        super().train()

        states = self.env.reset()

        for callback in self.callbacks:
            callback.on_before_train()

        for t in trange(self.timesteps):

            for callback in self.callbacks:
                callback.on_before_step(timestep=t)

            self.agent.on_before_step(timestep=t)

            # Get action tensor from each agent and stack them
            with torch.no_grad():
                notice.debug(f"States: {states}")
                for agent, state in zip(self.agents, states):
                    notice.debug(f"State: {state}, Agent: {agent}")
                actions = torch.vstack([agent.act(state) for agent, state in zip(self.agents, states)])
            next_states, rewards, terminated, truncated, info = self.env.step(actions)
            self.memory.add(
                {
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "next_states": next_states,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
            )

            with torch.no_grad():
                for agent in self.agents:
                    agent.collect(new_states, action, reward)

            with torch.no_grad():
                if terminated.any() or truncated.any():
                    states, infos = self.env.reset()

    def eval(self):
        super().eval()

    def save(self):
        pass
