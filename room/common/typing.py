from typing import Union

from room.agents.base import Agent
from room.policies import Policy, policies


class AgentWithPolicy(Agent, Policy):
    def __init__(self):
        pass
