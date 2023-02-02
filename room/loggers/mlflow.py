from pathlib import Path
from typing import Any, Dict, Optional, Union

from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

from room import notice
from room.common.utils import get_param, is_debug
from room.loggers.base import Logger, flatten_dict


class MLFlowLogger(Logger):

    EXTENSION = ".ckpt"

    def __init__(
        self,
        agent_id: int,
        cfg: Union[DictConfig, dict],
        exp_name: Optional[str] = None,
    ):
        super().__init__()

        self.id = agent_id
        self.tracking_uri = cfg["mlflow_uri"]
        self.client = MlflowClient(self.tracking_uri)
        self.cfg = cfg

        if cfg["debug"]:
            exp_name = "Debug"
        else:
            exp_name = get_param(exp_name, "exp_name", cfg, show=is_debug(cfg))

        self.experiment = self.client.get_experiment_by_name(exp_name)
        if self.experiment is None:
            self.experiment_id = self.client.create_experiment(exp_name)
            self.experiment = self.client.get_experiment(self.experiment_id)
        else:
            self.experiment_id = self.experiment.experiment_id

        self.run = self.client.create_run(self.experiment_id)
        self.run_id = self.run.info.run_id
        self.local_run_dir = (
            Path(".")
            / Path(self.tracking_uri.lstrip("file:"))
            / self.experiment_id
            / self.run_id
            / "artifacts"
        ).resolve()

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
                notice.warning(
                    f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}"
                )
                continue

            self.log_param(k, v)

    def log_artifact(self, local_path, artifact_path=None):
        self.client.log_artifact(self.run_id, local_path, artifact_path)
