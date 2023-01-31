from torch import nn

registered_criteria = {
    "huber": nn.HuberLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
    "kldiv": nn.KLDivLoss,
}
