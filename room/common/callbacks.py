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


class MLFlowLogging(MLFlowLogger, Callback):
    def __init__(self, tracking_uri, cfg, exp_name, *args, **kwargs):
        super().__init__(tracking_uri=tracking_uri, cfg=cfg, exp_name=exp_name, *args, **kwargs)
        notice.info("You can open the dashboard by `bash dashboard.sh`.")

    def on_timestep_start(self):
        self.log_hparams(self.cfg)

    def on_timestep_end(self):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self, *args, **kwargs):
        self.log_metrics(kwargs["metrics"], step=kwargs["timestep"])

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, agent, save_dir, top_k=1):
        self.agent = agent
        self.save_dir = save_dir
        self.top_ckpt = None
        self.top_total_reward = -1e10

    def on_timestep_start(self):
        pass

    def on_timestep_end(self):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self, *args, **kwargs):
        self.timestep = kwargs["timestep"]
        total_reward = kwargs["total_reward"]
        if total_reward > self.top_total_reward:
            self.top_total_reward = total_reward
            self.top_ckpt = self.agent.make_ckpt(self.timestep, total_reward)

    def on_train_start(self):
        pass

    def on_train_end(self):
        self.agent.save(ckpt=self.top_ckpt, path=self.save_dir / f"best_{self.timestep}.cpt")
