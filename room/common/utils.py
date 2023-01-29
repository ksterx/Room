from room import notice
from room.agents.base import Agent
from room.agents.policies import registered_policies


def check_agent(agent: Agent):
    if isinstance(agent.policy_name, str):

        if agent.policy not in registered_policies[agent.name]:
            raise ValueError(
                f"'{agent.policy_name}' is not available for {agent.name}. Available policies are {registered_policies[agent.name]}"
            )
