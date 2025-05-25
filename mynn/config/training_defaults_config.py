# mynn/config/training_defaults_config.py
from torch import device
from torch.cuda import is_available

# 训练设备
DEVICE = device("cuda" if is_available() else "cpu")

# 默认优化器参数
DEFAULT_OPTIMIZER_TYPE = "Adam"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9  # 主要用于 SGD
DEFAULT_WEIGHT_DECAY = 1e-4

# 默认训练参数
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 1
DEFAULT_LOSS_FUNCTION = "CrossEntropyLoss"  # 或者可选"MSELoss","L1Loss"
