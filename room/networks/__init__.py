from room.networks.blocks import MLP, MLPBN


def mlp3(state_shape, action_shape, hidden_dim=64):
    return MLP([state_shape, hidden_dim, hidden_dim, action_shape], activation="relu")


registered_models = {"mlp3": mlp3}
