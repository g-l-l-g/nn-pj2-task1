# mynn/config/experiments/dynamic_cnn_baseline_config.py

# 模型架构
ARCHITECTURE = [
    {'type': 'conv', 'params': {'out_channels': 32, 'kernel_size': 3, 'padding': 1}},
    {'type': 'bn',   'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},
    {'type': 'pool', 'params': {'kernel_size': 2, 'stride': 2}},

    {'type': 'conv', 'params': {'out_channels': 64, 'kernel_size': 3, 'padding': 1}},
    {'type': 'bn',   'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},
    {'type': 'pool', 'params': {'kernel_size': 2, 'stride': 2}},

    {'type': 'conv', 'params': {'out_channels': 128, 'kernel_size': 3, 'padding': 1}},
    {'type': 'bn',   'params': {}},
    {'type': 'relu', 'params': {'inplace': True}},
    {'type': 'pool', 'params': {'kernel_size': 2, 'stride': 2}},

    {'type': 'flatten', 'params': {}},
    {'type': 'fc',   'params': {'out_features': 128}},
    {'type': 'relu', 'params': {'inplace': True}},
    {'type': 'dropout', 'params': {}},  # p will be set by DynamicCNN from 'dropout_rate' in exp_config
    {'type': 'fc',   'params': {}}
]

# 实验配置
EXPERIMENT_CONFIG = {
    "model_type": "DynamicCNN",
    "architecture_name": "dynamic_cnn_baseline",
    "architecture_config": ARCHITECTURE,

    "optimizer_type": "Adam",
    "learning_rate": 0.001,
    "lr_scheduler_type": "MultiStepLR",
    "lr_scheduler_params": {"milestones": [100, 150],
                            "gamma": 0.1},

    "weight_decay": 1e-4,

    "num_epochs": 2,  # Kept low for quick testing, increase for real training
    "batch_size": 32,
    "loss_function": "CrossEntropyLoss",  # 可选"CrossEntropyLoss","MSELoss","L1Loss"
    "dropout_rate": 0.25,
}