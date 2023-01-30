from abc import ABC, abstractmethod

from room import notice
from room.loggers import MLFlowLogger


class Callback(ABC):
    @abstractmethod
    def on_timestep_start(self):
        raise NotImplementedError

    @abstractmethod
    def on_timestep_end(self):
        raise NotImplementedError

    @abstractmethod
    def on_episode_start(self):
        raise NotImplementedError

    @abstractmethod
    def on_episode_end(self):
        raise NotImplementedError

    @abstractmethod
    def on_train_start(self):
        raise NotImplementedError

    @abstractmethod
    def on_train_end(self):
        raise NotImplementedError


class MLFlowCallback(MLFlowLogger, Callback):
    def __init__(self, tracking_uri, cfg, exp_name, *args, **kwargs):
        super().__init__(tracking_uri=tracking_uri, cfg=cfg, exp_name=exp_name, *args, **kwargs)
        notice.info(f"You can open the dashboard by `bash dashboard.sh`.")

    def on_timestep_start(self):
        self.log_hparams(self.cfg)

    def on_timestep_end(self):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self, metircs, episode):
        self.log_metrics(metircs, episode)

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass
