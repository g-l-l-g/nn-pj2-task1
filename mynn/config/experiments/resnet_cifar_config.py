# mynn/config/experiments/resnet_cifar_config.py

INITIAL_CHANNELS = 64  # 定义初始通道数，便于在 ARCHITECTURE 中引用

ARCHITECTURE = [
    # 初始卷积块
    {'type': 'conv', 'params': {'out_channels': INITIAL_CHANNELS,
                                'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': False}},
    {'type': 'bn',   'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},

    # Stage 1: 2 个 ResBlock, 输出通道数 = INITIAL_CHANNELS
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS, 'stride': 1}},
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS, 'stride': 1}},

    # Stage 2: 2 个 ResBlock, 输出通道数 = INITIAL_CHANNELS * 2, 第一个 block 进行下采样
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 2, 'stride': 2}},
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 2, 'stride': 1}},

    # Stage 3: 2 个 ResBlock, 输出通道数 = INITIAL_CHANNELS * 4, 第一个 block 进行下采样
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 4, 'stride': 2}},
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 4, 'stride': 1}},

    # Stage 4: 2 个 ResBlock, 输出通道数 = INITIAL_CHANNELS * 8, 第一个 block 进行下采样
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 8, 'stride': 2}},
    {'type': 'resblock', 'params': {'out_channels': INITIAL_CHANNELS * 8, 'stride': 1}},

    # 分类器头部
    {'type': 'adaptiveavgpool', 'params': {'output_size': (1, 1)}},
    {'type': 'flatten', 'params': {}},
    {'type': 'fc',   'params': {}}  # 最后一个全连接层的 out_features 将被 DynamicCNN 自动设置为 num_classes
]

EXPERIMENT_CONFIG = {
    "model_type": "DynamicCNN", # Using DynamicCNN to build this ResNet-like arch
    "architecture_name": "resnet_cifar_cnn",
    "architecture_config": ARCHITECTURE,

    # 训练和优化器参数 (可以保持 ResNet 原有的设置)
    "optimizer_type": "SGD",
    "learning_rate": 0.1,
    "momentum": 0.9,

    "lr_scheduler_type": "MultiStepLR",
    "lr_scheduler_params": {"milestones": [100, 150],
                            "gamma": 0.1},

    "weight_decay": 1e-4,

    "num_epochs": 2,
    "batch_size": 32,
    "loss_function": "CrossEntropyLoss",  # 根据提供的文件，损失函数仍为 L1Loss
    "dropout_rate": 0,
}
