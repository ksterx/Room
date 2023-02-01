import glob
import os
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

    EXTENSION = ".ckpt"

    def __init__(self, agent, save_dir, tracking_uri, cfg, exp_name, top_k: int = 1, *args, **kwargs):
        super().__init__(tracking_uri=tracking_uri, cfg=cfg, exp_name=exp_name, *args, **kwargs)
        notice.info("You can open the dashboard by `bash dashboard.sh`.")

        self.agent = agent
        self.top_ckpt = None
        self.top_total_reward = -1e10

    def on_timestep_start(self):
        self.log_hparams(self.cfg)

    def on_timestep_end(self):
        pass

    def on_episode_start(self):
        pass

    def on_episode_end(self, *args, **kwargs):
        self.log_metrics(kwargs["metrics"], step=kwargs["episode"])
        self._save_ckpt(*args, **kwargs)

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def _save_ckpt(self, *args, **kwargs):
        # Save the checkpoint on every episode, in case of crash
        timestep = kwargs["timestep"]
        total_reward = kwargs["total_reward"]
        ckpt_path = self.local_run_dir / f"ckpt_{timestep}{self.EXTENSION}"
        best_ckpt_path = self.local_run_dir / f"best_{timestep}{self.EXTENSION}"
        tmp_ckpt = self.agent.make_ckpt(timestep, total_reward)
        for fn in glob.glob(str(self.local_run_dir / f"ckpt_*{self.EXTENSION}")):
            os.remove(fn)
        self.agent.save(ckpt=tmp_ckpt, path=ckpt_path)
        if total_reward > self.top_total_reward:
            self.top_total_reward = total_reward
            for fn in glob.glob(str(self.local_run_dir / f"best_*{self.EXTENSION}")):
                os.remove(fn)
            self.agent.save(ckpt=tmp_ckpt, path=best_ckpt_path)
            print(best_ckpt_path)
