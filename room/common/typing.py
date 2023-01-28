from typing import Dict, Union

from omegaconf import DictConfig

from room.agents.base import Agent
from room.policies import Policy, policies


class AgentWithPolicy(Agent, Policy):
    def __init__(self):
        pass


CfgType = Union[DictConfig, Dict]
PolicyType = Union[Policy, str]
