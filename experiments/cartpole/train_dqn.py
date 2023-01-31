import gym
import hydra
from kxmod.service import SlackBot
from omegaconf import DictConfig, OmegaConf

from room import notice
from room.agents import DQN
from room.common.callbacks import MLFlowCallback
from room.common.utils import flatten_dict
from room.envs import register_env
from room.trainers import SimpleTrainer


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(omegacfg: DictConfig) -> None:
    notice.info("Starting training...")
    if omegacfg.debug:
        notice.warning("Running in DEBUG mode")
    cfg = flatten_dict(OmegaConf.to_container(omegacfg, resolve=True))

    bot = SlackBot()
    mlf_callback = MLFlowCallback(omegacfg.mlflow_uri, cfg, omegacfg.exp_name)

    env = gym.make("CartPole-v1", render_mode="human")
    env = register_env(env)

    agent = DQN(model="mlp3", cfg=cfg)

    # mlf_logger = MLFlowLogger(omegacfg.mlflow_uri, omegacfg.exp_name, omegacfg)
    trainer = SimpleTrainer(env=env, agents=agent, memory="random", cfg=cfg, callbacks=[mlf_callback], **cfg)
    trainer.train()

    if not omegacfg.debug:
        bot.say("Training finished!")

    save_video = False
    if save_video:
        render_mode = "rgb_array"
    else:
        render_mode = "human"
    agent.play(
        gym.make("CartPole-v1", render_mode=render_mode),
        num_eps=5,
        save_video=save_video,
        save_dir="/tmp/experiments/cartpole",
    )
    if save_video:
        mlf_callback.client.log_artifacts(mlf_callback.run_id, "/tmp/experiments/cartpole")


if __name__ == "__main__":
    main()
