from typing import Any, Dict, Union

from omegaconf import DictConfig, OmegaConf
from room import logger
from room.train.core.loggers.logger import Logger, flatten_dict

from mlflow.tracking import MlflowClient


class MLFlowLogger(Logger):
    def __init__(self, tracking_uri: str, exp_name: str, cfg: DictConfig):
        super().__init__()
        self.client = MlflowClient(tracking_uri)

        if cfg.debug:
            exp_name = "Debug"

        self.experiment = self.client.get_experiment_by_name(exp_name)
        if self.experiment is None:
            self.experiment = self.client.create_experiment(exp_name)

        # convert hydra config to dict
        tags = OmegaConf.to_container(cfg.run, resolve=True)
        self.run_id = self.client.create_run(self.experiment.experiment_id, tags=tags).info.run_id
        self.log_hparams(cfg)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step: str = None):
        self.client.log_metric(self.run_id, key, value, step)

    def log_hparams(self, params: Union[Dict[str, Any], DictConfig]) -> None:

        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)

        params = flatten_dict(params)
        for k, v in params.items():
            if len(str(v)) > 250:
                logger.warning(f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}")
                continue

            self.log_param(k, v)

    def log_metrics(self, metrics):
        pass
