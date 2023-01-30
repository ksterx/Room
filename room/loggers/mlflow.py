from typing import Any, Dict, Optional, Union

from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from room import notice
from room.loggers.base import Logger, flatten_dict


class MLFlowLogger(Logger):
    def __init__(
        self,
        tracking_uri: str,
        cfg: Union[DictConfig, dict],
        exp_name: Optional[str] = None,
    ):
        super().__init__()
        self.client = MlflowClient(tracking_uri)
        self.cfg = cfg

        if cfg["debug"]:
            exp_name = "Debug"
        else:
            exp_name = cfg["exp_name"]

        self.experiment = self.client.get_experiment_by_name(exp_name)

        if self.experiment is None:
            self.experiment = self.client.create_experiment(exp_name)

        # convert hydra config to dict
        agent_tag = cfg["run"]
        self.run = self.client.create_run(self.experiment.experiment_id, tags={"agent": agent_tag})
        self.run_id = self.run.info.run_id
        # self.log_hparams(cfg)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value, step: Union[str, int]):
        self.client.log_metric(self.run_id, key, value, step)

    def log_metrics(self, metris: Dict[str, Any], step: Union[str, int]):
        for k, v in metris.items():
            self.log_metric(k, v, step)

    def log_hparams(self, params: Union[Dict[str, Any], DictConfig]) -> None:

        if isinstance(params, DictConfig):
            params = OmegaConf.to_container(params, resolve=True)
            params = flatten_dict(params)

        for k, v in params.items():
            if len(str(v)) > 250:
                notice.warning(f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}")
                continue

            self.log_param(k, v)
