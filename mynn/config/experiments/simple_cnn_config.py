# mynn/config/experiments/simple_cnn_config.py

# 模型架构 - 直接包含结构参数值
# 条件层 (BN, Dropout) 仍然可以通过 EXPERIMENT_CONFIG 中的标志控制
ARCHITECTURE = [
    # Block 1
    {'type': 'conv', 'params': {'out_channels': 32, 'kernel_size': 3, 'padding': 1}},  # filters_b1 = 32
    {'type': 'bn', 'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},  # activation_type_baseline = "ReLU"
    {'type': 'pool', 'params': {'kernel_size': 2, 'stride': 2}},

    # Block 2
    {'type': 'conv', 'params': {'out_channels': 64, 'kernel_size': 3, 'padding': 1}},  # filters_b2 = 64
    {'type': 'bn', 'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},

    {'type': 'pool', 'params': {'kernel_size': 2, 'stride': 2}},

    # Block 3
    {'type': 'conv', 'params': {'out_channels': 128, 'kernel_size': 3, 'padding': 1}},  # filters_b3 = 128
    {'type': 'bn', 'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},
    {'type': 'pool', 'params': {'kernel_size': 2, 'stride': 2}},

    # Fully Connected Layers
    {'type': 'flatten', 'params': {}},
    {'type': 'fc', 'params': {'out_features': 128}},  # fc_neurons_b = 128
    {'type': 'relu', 'params': {'inplace': True}},

    {'type': 'dropout', 'params': {}},
    {'type': 'fc', 'params': {}}  # 最终输出层
]

# 实验配置
EXPERIMENT_CONFIG = {
    "model_type": "DynamicCNN",
    "architecture_name": "simple_cnn",
    "architecture_config": ARCHITECTURE,

    "optimizer_type": "Adam",
    "lr_scheduler_type": "MultiStepLR",
    "lr_scheduler_params": {"milestones": [1],
                            "gamma": 0.1},
    "learning_rate": 0.001,
    "weight_decay": 1e-4,

    "num_epochs": 2,  # Kept low for quick testing, increase for real training
    "batch_size": 32,
    "loss_function": "CrossEntropyLoss",

    "dropout_rate": 0.0,
}
