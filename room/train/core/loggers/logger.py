from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def log_hparams(self, **kwargs):
        pass

    @abstractmethod
    def log_metrics(self, **kwargs):
        pass
