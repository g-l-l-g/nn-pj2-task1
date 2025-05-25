# mynn/config/experiments/__init__.py
from . import deep_resnet_config
from .dynamic_cnn_baseline_config import EXPERIMENT_CONFIG as dynamic_cnn_baseline_config
from .dynamic_cnn_deeper_config import EXPERIMENT_CONFIG as dynamic_cnn_deeper_config
from .resnet_cifar_config import EXPERIMENT_CONFIG as resnet_cifar_config
from .simple_cnn_config import EXPERIMENT_CONFIG as simple_cnn_config
from .deep_resnet_config import EXPERIMENT_CONFIG as deep_resnet_config

ALL_EXPERIMENT_CONFIGS = {
    "dynamic_cnn_baseline": dynamic_cnn_baseline_config,
    "dynamic_cnn_deeper": dynamic_cnn_deeper_config,
    "resnet_cifar": resnet_cifar_config,
    "simple_cnn": simple_cnn_config,
    "deep_resnet_config": deep_resnet_config,
}

__all__ = ["ALL_EXPERIMENT_CONFIGS"]
