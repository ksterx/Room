import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from room import logger
from room.envs import register_env
from room.train import MLFlowLogger, Trainer
from room.train.agents import PPO


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting training...")

    env = gym.make("CartPole-v1")
    env = register_env(env)
    mlf_logger = MLFlowLogger(cfg.mlflow_uri)

    agents = PPO()

    trainer = Trainer(env=env, agents=agents, cfg=cfg, logger=mlf_logger)
    trainer.train()


if __name__ == "__main__":
    main()
