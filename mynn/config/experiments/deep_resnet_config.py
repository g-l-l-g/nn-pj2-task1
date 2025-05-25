# mynn/config/experiments/new_deep_resnet_config.py

INITIAL_CHANNELS = 64  # 可以尝试增加这个值以获得更宽的网络，例如 96

# ResNet-34 like block configuration for CIFAR-10
# [3, 4, 6, 3] blocks for stages 1, 2, 3, 4 respectively
# Each 'resblock' typically has 2 conv layers. Total conv layers in blocks = 2*(3+4+6+3) = 32
# Plus the initial conv layer = 33 conv layers.
# This is a common adaptation of ResNet-34 for CIFAR-10.

RESNET_BLOCKS_STAGE1 = [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS, 'stride': 1}}
] * 3  # 3 blocks

RESNET_BLOCKS_STAGE2 = [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 2, 'stride': 2}}
] + [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 2, 'stride': 1}}
] * 3  # Remaining 3 blocks

RESNET_BLOCKS_STAGE3 = [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 4, 'stride': 2}}
] + [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 4, 'stride': 1}}
] * 5  # Remaining 5 blocks

RESNET_BLOCKS_STAGE4 = [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 8, 'stride': 2}}
] + [
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 8, 'stride': 1}}
] * 2  # Remaining 2 blocks


ARCHITECTURE = [
    {'type': 'conv', 'params': {'out_channels': INITIAL_CHANNELS,
                                'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}},
    {'type': 'bn',   'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},

    # Stages
    *RESNET_BLOCKS_STAGE1,
    *RESNET_BLOCKS_STAGE2,
    *RESNET_BLOCKS_STAGE3,
    *RESNET_BLOCKS_STAGE4,

    # 分类器头部
    {'type': 'adaptiveavgpool', 'params': {'output_size': (1, 1)}},
    {'type': 'flatten', 'params': {}},
    # Optional: Add Dropout before the final FC layer for deeper ResNets
    # {'type': 'dropout', 'params': {'p': 0.5}}, # Example
    {'type': 'fc',   'params': {}}
]

EXPERIMENT_CONFIG = {
    "model_type": "DynamicCNN",
    "architecture_name": "deep_resnet34_cifar_cnn", # New descriptive name
    "architecture_config": ARCHITECTURE,

    "optimizer_type": "SGD",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "num_epochs": 100,
    "batch_size": 128,
    "loss_function": "CrossEntropyLoss",
    "dropout_rate": 0,

    # --- Suggested additions for improved training ---
    "lr_scheduler_type": "CosineAnnealingLR",
    "lr_scheduler_params": {
        "T_max": 100,
        "eta_min": 0.001
    },
    # "label_smoothing": 0.1,
    # "use_mixup": True,
    # "mixup_alpha": 0.2
}