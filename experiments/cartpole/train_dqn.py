import pathlib
import tempfile

import gym
import hydra
from kxmod.service import SlackBot
from omegaconf import DictConfig, OmegaConf

from room import notice
from room.agents import DQN
from room.common.callbacks import MLFlowLogging
from room.common.utils import flatten_dict
from room.envs import register_env
from room.trainers import SimpleTrainer

ENV_NAME = "Acrobot-v1"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(omegacfg: DictConfig) -> None:

    if omegacfg.debug:
        notice.warning("Running in DEBUG mode")
    if omegacfg.render:
        render_mode = "human"
    else:
        render_mode = None

    cfg = flatten_dict(OmegaConf.to_container(omegacfg, resolve=True))

    env = gym.make(ENV_NAME, render_mode=render_mode)
    env = register_env(env)

    agent = DQN(model="mlp3", cfg=cfg)

    with tempfile.TemporaryDirectory() as d:
        artifact_dir = pathlib.Path(d)
        bot = SlackBot()
        mlf_logging = MLFlowLogging(
            agent=agent,
            save_dir=artifact_dir,
            tracking_uri=omegacfg.mlflow_uri,
            cfg=cfg,
            exp_name=omegacfg.exp_name,
        )

        trainer = SimpleTrainer(
            env=env,
            agents=agent,
            memory="random",
            cfg=cfg,
            callbacks=[mlf_logging],
            **cfg,
        )

        trainer.train()

        if not omegacfg.debug:
            bot.say("Training finished!")

        if omegacfg.save_video:
            render_mode = "rgb_array"
        else:
            render_mode = "human"
        agent.play(
            gym.make(ENV_NAME, render_mode=render_mode),
            num_eps=5,
            save_video=omegacfg.save_video,
            save_dir=artifact_dir,
        )

        mlf_logging.client.log_artifacts(mlf_logging.run_id, artifact_dir)

    notice.info("Program finished!")


if __name__ == "__main__":
    main()
