from typing import Union

import hydra
import wandb

# from models import create_model
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.monitor import Monitor

from room import logger
from room.training import agents, trainers


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    run_name = cfg.wandb_name
    cfg_dict = OmegaConf.to_container(cfg)
    if cfg.wandb_log:
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            name=run_name,
        )

    logger.info("Starting training...")

    trainer = Trainer(env_name=cfg.env_name, model_name=cfg.model_name)
    trainer.run()

    if cfg.wandb_log:
        wandb.finish()


if __name__ == "__main__":
    main()