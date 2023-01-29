from room.agents.policies.base import ActorCriticPolicy, Policy
from room.networks.blocks import MLP

registered_policies = {"a2c": ["mlp", "cnn"]}
policies = {"ac": ActorCriticPolicy}
