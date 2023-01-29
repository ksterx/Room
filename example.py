import gym
import hydra
from omegaconf import DictConfig, OmegaConf

from room import notice
from room.agents import DQN
from room.common.utils import flatten_dict
from room.envs import register_env
from room.loggers import MLFlowLogger
from room.memories import RandomMemory
from room.trainers import SequentialTrainer


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(omegacfg: DictConfig) -> None:
    notice.info("Starting training...")
    if omegacfg.debug:
        notice.warning("Running in DEBUG MODE")
    cfg = flatten_dict(OmegaConf.to_container(omegacfg, resolve=True))

    env = gym.make("CartPole-v1", render_mode="human")
    env = register_env(env)

    agent = DQN(model="mlp3", cfg=cfg)

    # mlf_logger = MLFlowLogger(omegacfg.mlflow_uri, omegacfg.exp_name, omegacfg)
    trainer = SequentialTrainer(env=env, agents=agent, memory="random", cfg=cfg, logger=None, **cfg)
    trainer.train()


if __name__ == "__main__":
    main()
