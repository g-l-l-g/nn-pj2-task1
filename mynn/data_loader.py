# mynn/data_loader.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
# 导入新的配置系统
from . import config as mynn_config


# 可选: 创建验证集分割的函数
def create_validation_split(train_set, val_split_ratio=0.1, random_seed=42):
    """
    从训练集中分割出一部分作为验证集。
    """
    if not (0 < val_split_ratio < 1):
        return train_set, None

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(val_split_ratio * num_train))

    rng = np.random.default_rng(random_seed)
    rng.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(train_set, val_idx)

    print(f"原始训练集大小: {num_train}")
    print(f"新训练子集大小: {len(train_subset)}")
    print(f"新验证子集大小: {len(val_subset)}")

    return train_subset, val_subset


def get_cifar10_loaders(batch_size_override=None, augment=True, val_split_ratio=None, random_seed=42, num_workers=2):
    """
    获取CIFAR-10的数据加载器。
    """
    batch_size_to_use = batch_size_override if batch_size_override is not None else mynn_config.DEFAULT_BATCH_SIZE

    base_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # CIFAR-10 mean/std for normalization to [-1, 1]
    ]

    if augment:
        train_transform = transforms.Compose([
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                             ] + base_transform_list)
    else:
        train_transform = transforms.Compose(base_transform_list)

    test_transform = transforms.Compose(base_transform_list)

    try:
        full_train_set = torchvision.datasets.CIFAR10(
            root=mynn_config.DATASET_ROOT_DIR,
            train=True,
            download=False,  # Set to True if not downloaded
            transform=train_transform
        )
        test_set = torchvision.datasets.CIFAR10(
            root=mynn_config.DATASET_ROOT_DIR,
            train=False,
            download=False,  # Set to True if not downloaded
            transform=test_transform
        )
    except RuntimeError as e:
        print(f"加载数据集 '{mynn_config.DATASET_ROOT_DIR}' 时出错: {e}")
        print(f"请确保 CIFAR-10 数据集 ('{mynn_config.CIFAR10_FOLDER_NAME}' 文件夹) 存在。")
        print("如果未下载，请在 torchvision.datasets.CIFAR10 中设置 download=True 并运行一次。")
        return None, None, None

    train_to_load = full_train_set
    val_loader_for_training = None

    if val_split_ratio and (0 < val_split_ratio < 1):
        train_subset, val_subset = create_validation_split(full_train_set, val_split_ratio, random_seed)
        if train_subset and val_subset:
            train_to_load = train_subset
            val_loader_for_training = DataLoader(
                val_subset, batch_size=batch_size_to_use, shuffle=False, num_workers=num_workers
            )
            print("使用从训练集分割出的验证集。")
        else:
            print("验证集分割失败或返回空子集。如果适用，将在训练循环中使用测试集进行验证。")
    else:
        print("未从训练集分割验证集。如果 valloader 是 test_loader，则将在训练循环中使用测试集进行验证。")

    train_loader_ = DataLoader(
        train_to_load, batch_size=batch_size_to_use, shuffle=True, num_workers=num_workers
    )
    test_loader_ = DataLoader(
        test_set, batch_size=batch_size_to_use, shuffle=False, num_workers=num_workers
    )

    effective_val_loader = val_loader_for_training if val_loader_for_training is not None else test_loader_

    if effective_val_loader and len(effective_val_loader.dataset) > 0:
        landscape_subset_indices = list(
            range(min(len(effective_val_loader.dataset), batch_size_to_use * 5)))  # e.g., 5 batches
        landscape_val_subset = Subset(effective_val_loader.dataset, landscape_subset_indices)
        landscape_val_loader = DataLoader(landscape_val_subset, batch_size=batch_size_to_use, shuffle=False,
                                          num_workers=num_workers)
    else:
        landscape_val_loader = None

    return train_loader_, effective_val_loader, test_loader_, landscape_val_loader
