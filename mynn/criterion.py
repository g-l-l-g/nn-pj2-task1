from . import config as mynn_config
import torch.nn as nn


def get_criterion(exp_config):
    loss_name = exp_config.get("loss_function", mynn_config.DEFAULT_LOSS_FUNCTION)
    loss_name_lower = loss_name.lower()
    if loss_name_lower == "crossentropyloss":
        return nn.CrossEntropyLoss()
    elif loss_name_lower == "mseloss":
        return nn.MSELoss()
    elif loss_name_lower == "l1loss":
        return nn.L1Loss()
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")