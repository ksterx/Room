from room.train.agents.agent import Agent
from room.train.policies import registered_policies


def check_agent(agent: Agent):
    if isinstance(agent.policy, str):
        if agent.policy not in registered_policies[agent.agent_name]:
            raise ValueError(
                f"{agent.policy} is not available for {agent.agent_name}. Available policies are {registered_policies[agent.agent_name]}"
            )
