# mynn/train.py
import time
import copy
import os
import json
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from . import config as mynn_config
from .models import DynamicCNN
from .utils.plot_training_results import plot_training_results
from .lr_scheduler import get_lr_scheduler


# 模型训练
def train_model(model, train_loader, val_loader, criterion, optimizer,
                full_experiment_config,
                num_epochs_override=None, device_override=None, exp_name="", output_dir=None,
                num_classes=None):

    num_epochs_to_use = num_epochs_override if num_epochs_override is not None else mynn_config.DEFAULT_NUM_EPOCHS
    device_to_use = device_override if device_override is not None else mynn_config.DEVICE

    if num_classes is None:
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        elif hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            if isinstance(model, nn.Sequential) and isinstance(model[-1], nn.Linear):
                num_classes = model[-1].out_features
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                num_classes = model.fc.out_features
            else:
                final_linear_layer = None
                if hasattr(model, 'features_sequence'):
                    for layer_module in reversed(model.features_sequence):
                        if isinstance(layer_module, nn.Linear):
                            final_linear_layer = layer_module
                            break
                if final_linear_layer:
                    num_classes = final_linear_layer.out_features
                else:  # Fallback
                    num_classes = len(mynn_config.CLASSES)
        print(f"训练时自动检测到类别数量: {num_classes}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    print(f"\n--- 为实验训练模型: {exp_name} ---")
    print(f"使用设备: {device_to_use}")
    print(f"Epoch 数量: {num_epochs_to_use}")
    model.to(device_to_use)

    if not output_dir:
        output_dir = os.path.join(mynn_config.RUNS_DIR_BASE, exp_name.replace(' ', '_'))
        print(f"警告: train_model 未提供 output_dir。回退到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- TensorBoard SummaryWriter ---
    tensorboard_log_dir = os.path.join(output_dir, 'tensorboard_logs')
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard 日志将保存在: {tensorboard_log_dir}")

    # --- 保存完整的实验配置 ---
    if full_experiment_config:
        config_save_path_json = os.path.join(output_dir, "full_experiment_config.json")
        try:
            serializable_config_for_json = {}
            for key, value in full_experiment_config.items():
                if isinstance(value, torch.device):
                    serializable_config_for_json[key] = str(value)
                else:
                    serializable_config_for_json[key] = value
            with open(config_save_path_json, 'w', encoding='utf-8') as f_json:
                json.dump(serializable_config_for_json, f_json, indent=4, ensure_ascii=False,
                          skipkeys=True)
        except TypeError as e_json:
            print(f"错误: 无法将实验配置序列化为 JSON: {e_json}")

    # TensorBoard记录训练过程的数据
    if train_loader and len(train_loader) > 0:
        try:
            sample_inputs, _ = next(iter(train_loader))
            writer.add_graph(model, sample_inputs.to(device_to_use))
            print("模型图已记录到 TensorBoard。")
        except Exception as e:
            print(f"记录模型图到 TensorBoard 时出错: {e}")

    # --- 初始化学习率调度器 ---
    scheduler = None
    if full_experiment_config:
        scheduler = get_lr_scheduler(optimizer, full_experiment_config, num_epochs_to_use)
        if scheduler:
            print(f"使用学习率调度器: {full_experiment_config.get('lr_scheduler_type')}")
    else:
        print("警告: 未提供 full_experiment_config，无法初始化学习率调度器。")

    # 模型开始训练
    since = time.time()
    for epoch in range(num_epochs_to_use):
        print(f'Epoch {epoch + 1}/{num_epochs_to_use}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                if val_loader is None:
                    val_losses.append(float('nan'))
                    val_accuracies.append(float('nan'))
                    if epoch > 0:
                        writer.add_scalar('Loss/val', float('nan'), epoch)
                        writer.add_scalar('Accuracy/val', float('nan'), epoch)
                    continue
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            if dataloader is None or len(dataloader) == 0:
                print(f"警告: 阶段 '{phase}' 的数据加载器为空或None。跳过此阶段 Epoch {epoch + 1}。")
                if phase == 'val':
                    val_losses.append(float('nan'))
                    val_accuracies.append(float('nan'))
                    writer.add_scalar('Loss/val', float('nan'), epoch)
                    writer.add_scalar('Accuracy/val', float('nan'), epoch)
                # Potentially log NaN for train phase as well if dataloader is empty
                elif phase == 'train':
                    train_losses.append(float('nan'))
                    train_accuracies.append(float('nan'))
                    writer.add_scalar('Loss/train', float('nan'), epoch)
                    writer.add_scalar('Accuracy/train', float('nan'), epoch)
                continue

            for inputs, original_int_labels in dataloader:
                inputs, original_int_labels = inputs.to(device_to_use), original_int_labels.to(device_to_use)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if isinstance(criterion, (nn.MSELoss, nn.L1Loss)):
                        target_labels = torch.nn.functional.one_hot(original_int_labels,
                                                                    num_classes=num_classes).float()
                    else:
                        target_labels = original_int_labels
                    loss = criterion(outputs, target_labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds == original_int_labels.data).to(torch.int)).item()
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples if total_samples > 0 else float('nan')
            epoch_acc = running_corrects / total_samples if total_samples > 0 else float('nan')
            print(f'{phase} 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:  # phase == 'val'
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)
                if not torch.isnan(torch.tensor(epoch_acc)) and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_model_filename = mynn_config.DEFAULT_BEST_MODEL_SAVE_FILENAME.replace(
                        ".pth", f"_{exp_name.replace(' ', '_')}.pth"
                    )
                    save_path = os.path.join(output_dir, best_model_filename)
                    torch.save(model.state_dict(), save_path)
                    print(f"最佳验证准确率提升至 {best_acc:.4f}，模型保存至 {save_path}")

        # 为加快训练速度可以不保存中间过程的权重，减少内存占用，将以下代码注释
        # torch.save(model.state_dict(), os.path.join(output_dir, f"epoch_{epoch + 1}.pth"))

        # --- 在每个 epoch 结束后更新学习率 ---
        if scheduler:
            current_lr_before_step = optimizer.param_groups[0]['lr']
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                current_val_loss_for_scheduler = val_losses[-1] if val_losses and not torch.isnan(
                    torch.tensor(val_losses[-1])) else None
                if current_val_loss_for_scheduler is not None:
                    scheduler.step(current_val_loss_for_scheduler)
                else:
                    print(
                        f"警告: Epoch {epoch + 1} - ReduceLROnPlateau 需要有效的验证损失才能执行 step。当前验证损失无效。")
            else:
                scheduler.step()

            current_lr_after_step = optimizer.param_groups[0]['lr']
            if abs(current_lr_before_step - current_lr_after_step) > 1e-7:  # 比较浮点数
                print(
                    f"学习率在 epoch {epoch + 1} 结束时从 {current_lr_before_step:.6e} 更新为 {current_lr_after_step:.6e}")
            writer.add_scalar('LearningRate', current_lr_after_step, epoch)

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳验证准确率: {best_acc:.4f}')

    writer.add_hparams(
        {k: str(v)[:250] for k, v in full_experiment_config.items() if isinstance(v, (str, int, float, bool))},
        # Log hyperparams
        {'hparam/best_val_accuracy': best_acc,
         'hparam/final_train_accuracy': train_accuracies[-1] if train_accuracies and not torch.isnan(
             torch.tensor(train_accuracies[-1])) else -1,
         'hparam/final_val_accuracy': val_accuracies[-1] if val_accuracies and not torch.isnan(
             torch.tensor(val_accuracies[-1])) else -1
         }
    )

    if best_acc > 0 or any(
            va is not None and not torch.isnan(torch.tensor(va)) for va in val_accuracies if va is not None):
        model.load_state_dict(best_model_wts)
    else:
        print("验证准确率无提升，或跳过了验证。使用最后一个epoch的权重。")

    valid_tl = [l for l in train_losses if l is not None and not torch.isnan(torch.tensor(l))]
    valid_vl = [l for l in val_losses if l is not None and not torch.isnan(torch.tensor(l))]
    valid_ta = [a for a in train_accuracies if a is not None and not torch.isnan(torch.tensor(a))]
    valid_va = [a for a in val_accuracies if a is not None and not torch.isnan(torch.tensor(a))]

    if valid_tl and valid_ta:
        plot_training_results(valid_tl, valid_vl, valid_ta, valid_va, exp_name, output_dir=output_dir)
    else:
        print("跳过绘制训练结果，因为历史列表为空或包含无效值。")

    writer.close()

    return model, {"train_loss": train_losses, "val_loss": val_losses,
                   "train_acc": train_accuracies, "val_acc": val_accuracies,
                   "best_val_acc": best_acc.item() if isinstance(best_acc, torch.Tensor) else best_acc}
