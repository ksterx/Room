from typing import Any, Dict

from room import logger
from room.train.core.loggers.logger import Logger

from mlflow.tracking import MlflowClient


class MLFlowLogger(Logger):
    def __init__(self, tracking_uri):
        super().__init__()
        logger.info("Initializing MLFlowLogger")
        logger.debug(f"tracking_uri: {tracking_uri}")
        self.client = MlflowClient(tracking_uri)

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
