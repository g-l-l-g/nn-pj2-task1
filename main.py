import torch
import os
import torch.nn as nn
import numpy as np
import copy  # ADDED for model creator deepcopy
from datetime import datetime

from mynn import config as mynn_config
from mynn.data_loader import get_cifar10_loaders
from mynn.models import DynamicCNN, ResBlock
from mynn.train import train_model
from mynn.criterion import get_criterion
from mynn.optimizer import get_optimizer
from mynn.evaluate import evaluate_model
from mynn.utils.image_show import imshow


def run_experiment(exp_name, exp_dir):
    print(f"\n======================================================")
    print(f" 开始实验: {exp_name} ")
    print(f"======================================================")

    if exp_name not in mynn_config.ALL_EXPERIMENT_CONFIGS:
        print(f"错误: 实验配置 '{exp_name}' 未找到。")
        return None
    exp_config = mynn_config.ALL_EXPERIMENT_CONFIGS[exp_name]

    current_exp_config = copy.deepcopy(exp_config)

    print(f"{exp_name} 的配置:")
    print(f"  模型类型: {current_exp_config.get('model_type', 'N/A')}")
    print(f"  优化器: {current_exp_config.get('optimizer_type', mynn_config.DEFAULT_OPTIMIZER_TYPE)}")
    print(f"  学习率: {current_exp_config.get('learning_rate', mynn_config.DEFAULT_LEARNING_RATE)}")
    print(f"  损失函数: {current_exp_config.get('loss_function', mynn_config.DEFAULT_LOSS_FUNCTION)}")

    safe_exp_name = exp_name.replace(' ', '_').replace('/', '_')
    exp_run_dir = os.path.join(exp_dir, safe_exp_name)
    try:
        os.makedirs(exp_run_dir, exist_ok=True)
        print(f"实验目录: {exp_run_dir}")
    except OSError as e:
        print(f"严重错误: 无法创建实验目录 {exp_run_dir}: {e}。中止。")
        return {"error": f"目录创建失败: {exp_run_dir}"}

    print("\n[阶段 1] 加载数据...")
    batch_size = current_exp_config.get("batch_size", mynn_config.DEFAULT_BATCH_SIZE)

    train_loader, val_loader, test_loader_final, landscape_val_loader = get_cifar10_loaders(
        batch_size_override=batch_size, augment=True,
        val_split_ratio=0.1, num_workers=2)
    if not train_loader or not test_loader_final:
        print("加载数据失败 (训练或测试加载器缺失)。中止实验。")
        return {"error": "数据加载失败"}
    if not landscape_val_loader:
        print("警告: 未能创建用于损失函数地形图的验证子集加载器。将跳过该步骤。")
    print("数据加载成功。")

    print("\n[阶段 2] 初始化模型...")
    model_type = current_exp_config.get("model_type")
    if not model_type:
        print(f"错误: 实验 '{exp_name}' 未定义 'model_type'。中止。")
        return {"error": "model_type 未定义"}

    model_instance = None
    num_classes = len(mynn_config.CLASSES)  # Standard CIFAR-10 classes

    if model_type == "DynamicCNN":
        arch_config = current_exp_config.get("architecture_config")
        if not arch_config:
            print(f"错误: DynamicCNN 实验 '{exp_name}' 未定义 'architecture_config'。中止。")
            return {"error": "DynamicCNN 缺少架构"}
        model_instance = DynamicCNN(num_classes=num_classes, architecture_config=arch_config,
                                    exp_config=current_exp_config)
    else:
        print(f"不支持的模型类型: {model_type} (main.py 中未直接处理，依赖DynamicCNN的灵活性)。中止。")
        return {"error": f"不支持的模型类型: {model_type}"}

    print(f"使用模型: {model_type}")
    total_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
    print(f"总可训练参数: {total_params:,}")

    initial_weights_state_dict = copy.deepcopy(model_instance.state_dict())
    initial_weights_path = (
        os.path.join(exp_run_dir,
                     mynn_config.DEFAULT_BEST_MODEL_SAVE_FILENAME.replace(".pth", f"_{safe_exp_name}.pth")
                     )
    )
    torch.save(initial_weights_state_dict, initial_weights_path)
    print(f"初始模型权重已保存至: {initial_weights_path}")  # Less verbose, or save only if needed

    model_instance.to(mynn_config.DEVICE)
    print("模型已初始化并移至设备。")

    print("\n[阶段 3] 定义损失函数和优化器...")
    criterion_train = get_criterion(current_exp_config)
    optimizer_train = get_optimizer(model_instance, current_exp_config)
    print("损失函数和优化器已定义。")

    print("\n[阶段 4] 训练模型...")
    num_epochs = current_exp_config.get("num_epochs", mynn_config.DEFAULT_NUM_EPOCHS)

    trained_model, history = train_model(model_instance, train_loader, val_loader, criterion_train, optimizer_train,
                                         num_epochs_override=num_epochs, device_override=mynn_config.DEVICE,
                                         full_experiment_config=exp_config,  # Pass the original config for saving
                                         exp_name=safe_exp_name, output_dir=exp_run_dir, num_classes=num_classes)
    print("训练完成。")
    print(f"TensorBoard 日志位于: {os.path.join(exp_run_dir, 'tensorboard_logs')}")
    best_val_acc_train = history.get('best_val_acc', 0.0) if history else 0.0

    print("\n[阶段 5] 在最终测试集上评估模型...")
    trained_model.eval()
    criterion_eval = get_criterion(current_exp_config)
    test_loss, test_acc, test_error, num_classes_detected_eval = evaluate_model(trained_model, test_loader_final,
                                                                                criterion_eval,
                                                                                device_override=mynn_config.DEVICE,
                                                                                exp_name=safe_exp_name,
                                                                                output_dir=exp_run_dir,
                                                                                num_classes=num_classes)
    print(f"{exp_name} 在测试集上的评估完成: 准确率={test_acc:.4f}, 损失={test_loss:.4f}")

    return {"experiment_name": exp_name, "model_type": model_type, "total_params": total_params,
            "best_validation_accuracy": best_val_acc_train, "test_accuracy": test_acc,
            "test_error": test_error, "test_loss": test_loss, "run_directory": exp_run_dir}


if __name__ == '__main__':
    results_summary = []
    experiments_to_run = ["deep_resnet_config"]
    # experiments_to_run = ["dynamic_cnn_baseline", "simple_cnn"] # For testing multiple
    experiment_dir = os.path.join(mynn_config.RUNS_DIR_BASE, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    for exp_name_run in experiments_to_run:
        result = run_experiment(exp_name_run, experiment_dir)
        if result and "error" not in result:
            results_summary.append(result)
        elif result and "error" in result:
            print(f"实验 {exp_name_run} 因错误失败: {result['error']}")
        elif not result:
            print(f"实验 {exp_name_run} 未成功完成。")

    print("\n\n================ 总体结果摘要 ================")
    if not results_summary:
        print("没有实验运行或成功完成以供摘要。")
    else:
        results_summary.sort(
            key=lambda x: x.get("test_accuracy", 0.0) if isinstance(x.get("test_accuracy"), float) else 0.0,
            reverse=True)

        for idx, res_item in enumerate(results_summary):
            print(f"\n--- 结果 {idx + 1} ---")
            print(f"实验: {res_item['experiment_name']}")
            print(f"  模型类型: {res_item['model_type']}")
            print(f"  总参数量: {res_item.get('total_params', 'N/A'):,}")

            best_val_acc = res_item.get('best_validation_accuracy', 'N/A')
            print(f"  最佳验证准确率: {best_val_acc:.4f}" if isinstance(best_val_acc, float) else f"最佳验证准确率: {best_val_acc}")

            test_acc_res = res_item.get('test_accuracy', 'N/A')
            print(f"  测试准确率: {test_acc_res:.4f}" if isinstance(test_acc_res, float) else f"测试准确率: {test_acc_res}")

            test_err_res = res_item.get('test_error', 'N/A')
            print(f"  测试错误率: {test_err_res:.4f}" if isinstance(test_err_res, float) else f"测试错误率: {test_err_res}")

            test_loss_res = res_item.get('test_loss', 'N/A')
            print(f"  测试损失: {test_loss_res:.4f}" if isinstance(test_loss_res, float) else f"测试损失: {test_loss_res}")

            print(f"  运行目录: {res_item.get('run_directory', 'N/A')}")
            print(f"  TensorBoard 日志: {os.path.join(res_item.get('run_directory', 'N/A'), 'tensorboard_logs')}")

        if results_summary and isinstance(results_summary[0].get("test_accuracy"), float):
            best_exp_summary = results_summary[0]
            print(f"\n--- 表现最佳的实验 (按测试准确率) ---")
            print(f"名称: {best_exp_summary['experiment_name']}")
            print(f"模型类型: {best_exp_summary['model_type']}")
            print(f"测试准确率: {best_exp_summary.get('test_accuracy', 'N/A'):.4f}")
            print(f"结果保存在: {best_exp_summary.get('run_directory', 'N/A')}")
