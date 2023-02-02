from typing import List, Optional, Union

import torch
from kxmod.service import SlackBot
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


class SimpleTrainer(Trainer):
    def __init__(
        self,
        env: EnvWrapper,
        agents: Agent,
        memory: Optional[Union[str, Memory]] = None,
        batch_size: Optional[int] = None,
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
            batch_size=batch_size,
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

        episode = 0
        total_reward = 0
        states, info = self.env.reset()

        self.on_train_start()

        # Training loop
        for t in trange(self.timesteps):

            self.on_timestep_start()

            # Get action tensor from each agent and stack them
            with torch.no_grad():
                actions = torch.vstack(
                    [agent.act(state, step=t) for agent, state in zip(self.agents, states)]
                ).to(self.device)
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
            total_reward += rewards.sum().item()  # TODO: check if this is correct

            self.on_timestep_end()

            states = next_states  # TODO: !!!!!!!!!!!!!!!!!!!!!!!!FIX HERE !!!!!!!!!!!!!!!!!!!!!!

            with torch.no_grad():
                if terminated.any() or truncated.any():
                    metrics = {"total_reward": total_reward}
                    monitor = {
                        "metrics": metrics,
                        "episode": episode,
                        "timestep": t,
                        "total_reward": total_reward,
                    }

                    self.on_episode_end(**monitor)

                    states, _ = self.env.reset()
                    episode += 1
                    total_reward = 0

            if len(self.memory) <= self.batch_size:
                continue
            else:
                batch = self.memory.sample(batch_size=self.batch_size)
                for agent in self.agents:
                    agent.learn(batch)

        self.on_train_end()

    def eval(self):
        super().eval()
