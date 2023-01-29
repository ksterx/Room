from typing import List, Optional, Union

import torch
from torch import optim
from tqdm import trange

from room import notice
from room.agents import Agent
from room.common.callbacks import Callback
from room.common.utils import get_param
from room.envs.wrappers import EnvWrapper
from room.loggers import Logger
from room.memories import Memory
from room.trainers.base import Trainer


class SequentialTrainer(Trainer):
    def __init__(
        self,
        env: EnvWrapper,
        agents: Union[Agent, List[Agent]],
        memory: Optional[Union[str, Memory]] = None,
        optimizer: Optional[Union[str, optim.Optimizer]] = None,
        timesteps: Optional[int] = None,
        device: Optional[Union[torch.device, str]] = None,
        logger: Optional[Logger] = None,
        cfg: dict = None,
        callbacks: Union[Callback, List[Callback]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            env=env,
            agents=agents,
            memory=memory,
            optimizer=optimizer,
            timesteps=timesteps,
            device=device,
            logger=logger,
            cfg=cfg,
            callbacks=callbacks,
            *args,
            **kwargs,
        )

    def train(self):
        super().train()

        states = self.env.reset()

        self.on_before_train()

        for t in trange(self.timesteps):

            for callback in self.callbacks:
                callback.on_before_step(timestep=t)

            for agent in self.agents:
                agent.on_before_step(timestep=t)

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
