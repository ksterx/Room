from abc import ABC, abstractmethod
from typing import List, Union

from room import logger
from room.env.wrappers import EnvWrapper

from ..agent import Agent


class Trainer(ABC):
    def __init__(self, env: EnvWrapper, agents: Union[Agent, List[Agent]], cfg):
        """Reinforcement learning trainer.

        Args:
            env (Env): environment to train on
            agents (Union[Agent, List[Agent]]): Agent(s) to train.
                If agents are being trained simultaneously, they should be passed as a list.
            cfg (yaml): Configurations
        """
        self.env = env
        self.agents = agents
        self.cfg = cfg

        # Set up agents
        self.num_agents = 0

    @property
    def num_agents(self) -> int:
        if type(self.agents) in [tuple, list]:
            return len(self.agents)
        else:
            return 1

    @abstractmethod
    def train(self):

        # Set up experiment logger
        if not self.cfg.log_metrics:
            logger.warning("Experiment logger is not active")
        elif self.cfg.log_metrics and not self.cfg.debug:
            logger.info("Experiment logger is active")
            import mlflow

            mlflow.start_run(run_name=self.cfg.run_name)

    @abstractmethod
    def eval(self):
        logger.warning("Use SequentialTrainer or ParallelTrainer instead of Trainer")
        quit()

    @abstractmethod
    def save(self):
        logger.warning("Use SequentialTrainer or ParallelTrainer instead of Trainer")
        quit()
