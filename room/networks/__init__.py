from room.networks.blocks import MLP, MLPBN


def mlp3(state_dim, action_dim, hidden_dim=64):
    return MLP([state_dim, hidden_dim, hidden_dim, action_dim], activation="relu")


registered_models = {"mlp3": mlp3}
