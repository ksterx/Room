import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from room.agents import Agent
from room.common.utils import flatten_dict
from room.loggers import MLFlowLogger
from room.trainers import OffPolicyTrainer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(omegacfg: DictConfig) -> None:

    ray.init(local_mode=omegacfg.debug)

    trainer = OffPolicyTrainer(device=0, logger=MLFlowLogger)
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()