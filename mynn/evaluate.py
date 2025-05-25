import torch
from tqdm import tqdm
import os
import numpy as np
import torch.nn as nn  # 需要导入 nn
from . import config as mynn_config
from .utils.plot_confusion_matrix import plot_confusion_matrix


def evaluate_model(model, testloader, criterion, device_override=None, exp_name="", output_dir=None,
                   num_classes=None):
    device_to_use = device_override if device_override is not None else mynn_config.DEVICE
    model.to(device_to_use)
    model.eval()
    running_loss, running_corrects, total_samples = 0.0, 0, 0
    all_preds, all_labels_original_int = [], []  # 存储原始整数标签

    if num_classes is None:
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        elif hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):  # For generic models
            # Heuristic: if final layer is Linear, its out_features is num_classes
            if isinstance(model, nn.Sequential) and isinstance(model[-1], nn.Linear):
                num_classes = model[-1].out_features
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                num_classes = model.fc.out_features
            else:  # Try to find the last linear layer for DynamicCNN
                final_linear_layer = None
                if hasattr(model, 'features_sequence'):  # DynamicCNN
                    for layer in reversed(model.features_sequence):
                        if isinstance(layer, nn.Linear):
                            final_linear_layer = layer
                            break
                if final_linear_layer:
                    num_classes = final_linear_layer.out_features
                else:
                    num_classes = len(mynn_config.CLASSES)  # Fallback
        else:
            num_classes = len(mynn_config.CLASSES)
        # print(f"评估时自动检测到类别数量: {num_classes}")

    # print(f"\n--- 为实验评估模型: {exp_name} 在测试集上 ---")
    if testloader is None or len(testloader) == 0:
        print("警告: Testloader 为空或None。无法评估。")
        return 0.0, 0.0, 1.0, 0  # Added num_classes_detected

    with torch.no_grad():
        # loader_to_iterate = tqdm(testloader, desc=f"评估 {exp_name}", leave=False)
        for inputs, original_int_labels_batch in testloader:
            # Removed tqdm for cleaner output when called by loss landscape
            inputs, original_int_labels_batch = inputs.to(device_to_use), original_int_labels_batch.to(device_to_use)
            outputs = model(inputs)

            if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
                target_labels_batch = torch.nn.functional.one_hot(original_int_labels_batch,
                                                                  num_classes=num_classes).float()
            else:
                target_labels_batch = original_int_labels_batch

            loss = criterion(outputs, target_labels_batch)
            _, preds_batch = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum((preds_batch == original_int_labels_batch.data).to(torch.int)).item()
            total_samples += inputs.size(0)
            all_preds.extend(preds_batch.cpu().numpy())
            all_labels_original_int.extend(original_int_labels_batch.cpu().numpy())  # 存储原始标签

    if total_samples == 0:
        # print("警告: 迭代后在 testloader 中未找到样本。无法评估。")
        return 0.0, 0.0, 1.0, num_classes

    test_loss_ = running_loss / total_samples
    test_acc_ = running_corrects / total_samples
    test_error_ = 1.0 - test_acc_
    # print(f'测试损失: {test_loss_:.4f}')
    # print(f'测试准确率: {test_acc_:.4f} (正确: {running_corrects}/{total_samples})')
    # print(f'测试错误率: {test_error_:.4f}')

    if output_dir:
        print(f"\n--- 为实验评估模型: {exp_name} 在测试集上 ---")
        print(f'测试损失: {test_loss_:.4f}')
        print(f'测试准确率: {test_acc_:.4f} (正确: {running_corrects}/{total_samples})')
        print(f'测试错误率: {test_error_:.4f}')
        class_correct = list(0. for _ in range(num_classes))
        class_total = list(0. for _ in range(num_classes))

        all_preds_tensor = torch.tensor(all_preds, device='cpu')
        all_labels_tensor = torch.tensor(all_labels_original_int, device='cpu')

        if all_labels_tensor.numel() > 0 and all_preds_tensor.numel() > 0:
            correct_predictions = (all_preds_tensor == all_labels_tensor)
            for i in range(len(all_labels_tensor)):
                label = all_labels_tensor[i].item()
                if 0 <= label < num_classes:
                    class_correct[label] += correct_predictions[i].item()
                    class_total[label] += 1
                else:
                    print(f"警告: 遇到越界标签 {label} 在索引 {i}。最大类别索引: {num_classes - 1}")
            print("\n逐类准确率:")
            for i in range(num_classes):
                if i < len(mynn_config.CLASSES):  # Check bounds for class name
                    class_name_display = mynn_config.CLASSES[i]
                else:
                    class_name_display = f"Class {i}"

                if class_total[i] > 0:
                    print(
                        f'准确率 {class_name_display:<10s} : {100 * class_correct[i] / class_total[i]:.2f}% '
                        f'({int(class_correct[i])}/{int(class_total[i])})')
                else:
                    print(f'准确率 {class_name_display:<10s} : N/A (0 样本)')
        else:
            print("警告: 标签或预测张量为空，跳过逐类准确率。")

        os.makedirs(output_dir, exist_ok=True)
        # Use all classes if num_classes > len(mynn_config.CLASSES), otherwise use mynn_config.CLASSES
        class_names_for_plot = mynn_config.CLASSES if num_classes <= len(mynn_config.CLASSES) else [f"Class {i}" for i
                                                                                                    in
                                                                                                    range(num_classes)]
        plot_confusion_matrix(all_preds, all_labels_original_int, class_names_for_plot, exp_name,
                              output_dir=output_dir)

    return test_loss_, test_acc_, test_error_, num_classes
