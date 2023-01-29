from room import notice
from room.networks.blocks import MLP, MLPBN


def test_mlp():
    mlp = MLP(layers=[1, 20, 30], activation="relu", output_activation="softmax")
