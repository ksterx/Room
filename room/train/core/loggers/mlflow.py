from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf
from room import logger
from room.train.core.loggers.logger import Logger

from mlflow.tracking import MlflowClient


class MLFlowLogger(Logger):
    def __init__(self, tracking_uri: str, exp_name: str, cfg: DictConfig):
        super().__init__()
        self.client = MlflowClient(tracking_uri)

        if not self.client.get_experiment_by_name(exp_name):
            self.client.create_experiment(exp_name)
        exp = self.client.get_experiment_by_name(exp_name)
        # convert hydra config to dict
        tags = OmegaConf.to_container(cfg.run, resolve=True)
        self.run = self.client.create_run(exp.experiment_id, tags=tags)

    def log_hparams(self, params: Dict[str, Any]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        for k, v in params.items():
            if len(str(v)) > 250:
                rank_zero_warn(
                    f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}", category=RuntimeWarning
                )
                continue

            self.experiment.log_param(self.run_id, k, v)

    def log_metrics(self, metrics):
        pass
