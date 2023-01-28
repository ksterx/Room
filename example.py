import gym
import hydra
from omegaconf import DictConfig

from room import notice
from room.agents import A2C
from room.common.utils import check_agent
from room.core import MLFlowLogger, SequentialTrainer
from room.envs import register_env


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    notice.info("Starting training...")
    if cfg.debug:
        notice.warning("Running in DEBUG MODE")

    env = gym.make("CartPole-v1", render_mode="human")
    env = register_env(env)

    agent1 = A2C(policy="ac", cfg=cfg)
    check_agent(agent1)
    agents = [agent1]

    mlf_logger = MLFlowLogger(cfg.mlflow_uri, cfg.exp_name, cfg)
    trainer = SequentialTrainer(env=env, agents=agents, cfg=cfg, logger=mlf_logger)
    trainer.train(cfg.agent.max_timesteps)


if __name__ == "__main__":
    main()
