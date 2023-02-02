import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from room.agents import Agent
from room.common.callbacks import MLFlowLogging
from room.common.utils import flatten_dict
from room.trainers import OffPolicyTrainer


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(omegacfg: DictConfig) -> None:

    ray.init(local_mode=omegacfg.debug)

    cfg = flatten_dict(OmegaConf.to_container(omegacfg, resolve=True))

    mlf_logging = MLFlowLogging(
        tracking_uri=omegacfg.mlflow_uri, cfg=cfg, exp_name=omegacfg.exp_name
    )

    trainer = OffPolicyTrainer(cfg=cfg, callbacks=[mlf_logging])
    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()
