import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from room import logger
from room.env import register_env
from room.train import Trainer
from room.train.agent import PPO


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting training...")

    env = gym.make("CartPole-v1")
    env = register_env(env)

    agents = PPO()

    trainer = Trainer(env=env, agents=agents, cfg=cfg)
    trainer.run()


if __name__ == "__main__":
    main()
