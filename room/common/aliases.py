from torch import nn, optim

registered_criteria = {
    "huber": nn.HuberLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "kldiv": nn.KLDivLoss,
}

registered_optimizers = {
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
    "sgd": optim.SGD,
    "adagrad": optim.Adagrad,
}
