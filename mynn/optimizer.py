import torch.optim as optim
from . import config as mynn_config


def get_optimizer(model, exp_config):
    optimizer_name = exp_config.get("optimizer_type", mynn_config.DEFAULT_OPTIMIZER_TYPE)
    lr = exp_config.get("learning_rate", mynn_config.DEFAULT_LEARNING_RATE)
    weight_decay = exp_config.get("weight_decay", mynn_config.DEFAULT_WEIGHT_DECAY)

    if optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        momentum = exp_config.get("momentum", mynn_config.DEFAULT_MOMENTUM)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")
