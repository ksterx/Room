from abc import ABC, abstractmethod


class Distribution(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_prob(self, *args, **kwargs):
        pass

    @abstractmethod
    def entropy(self, *args, **kwargs):
        pass

    @abstractmethod
    def mode(self, *args, **kwargs):
        pass

    @abstractmethod
    def mean(self, *args, **kwargs):
        pass

    @abstractmethod
    def variance(self, *args, **kwargs):
        pass

    @abstractmethod
    def kl(self, *args, **kwargs):
        pass

    @abstractmethod
    def cross_entropy(self, *args, **kwargs):
        pass

    @abstractmethod
    def entropy_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def cross_entropy_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def kl_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_prob_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def mode_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def mean_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def variance_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_entropy(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_kl(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_cross_entropy(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_log_prob(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_mean(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_variance(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_mode(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_entropy_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample_kl_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def
