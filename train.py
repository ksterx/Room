import gym
import hydra
from omegaconf import DictConfig

from room import logger
from room.envs import register_env
from room.train import MLFlowLogger, Trainer
from room.train.agents import A2C


# TODO: change hydra default output directory
@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting training...")
    if cfg.debug:
        logger.warning("Running in DEBUG MODE")

    env = gym.make("CartPole-v1")
    env = register_env(env)

    # agents = A2C()

    mlf_logger = MLFlowLogger(cfg.mlflow_uri, cfg.exp_name, cfg)
    trainer = Trainer(env=env, agents=agents, cfg=cfg, logger=mlf_logger)
    trainer.train()


if __name__ == "__main__":
    main()
