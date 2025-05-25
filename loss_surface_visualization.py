# 3d_loss_surface_visualization.py
import torch
import os
import json
import sys
import copy

from mynn.models import DynamicCNN
from mynn.data_loader import get_cifar10_loaders
from mynn.criterion import get_criterion
from mynn.utils.plot_3d_loss_surface import plot_3d_loss_surface_plotly
from mynn import config as mynn_base_config


def main_visualize_3d_loss(config_path, experiment_root_path, n_points, range_scale_alpha, range_scale_beta,
                           num_batches_for_loss_surface, output_base_dir="visualizations/3d_loss_surface"):
    """
    Generates and saves an interactive 3D loss surface plot.
    """
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            exp_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path}")
        return

    num_epochs = exp_config["num_epochs"]
    model_type = exp_config.get("model_type")
    architecture_config = exp_config.get("architecture_config")
    exp_name = exp_config.get("architecture_name", "unknown_experiment")
    batch_size = exp_config.get("batch_size", 32)

    safe_exp_name = exp_name.replace(' ', '_').replace('/', '_')
    output_dir = os.path.join(output_base_dir, safe_exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"3D Loss Surface visualizations will be saved to: {output_dir}")

    device = torch.device(mynn_base_config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data (Validation or Test Loader for landscape)
    print("\nLoading data for loss surface...")

    _, val_loader, test_loader_final, landscape_val_loader = get_cifar10_loaders(
        batch_size_override=batch_size, augment=False, val_split_ratio=0.1
    )

    loader_for_landscape = landscape_val_loader
    if not loader_for_landscape or len(loader_for_landscape.dataset) == 0:
        print("Warning: landscape_val_loader is not suitable, trying val_loader.")
        loader_for_landscape = val_loader
    if not loader_for_landscape or len(loader_for_landscape.dataset) == 0:
        print("Warning: val_loader is not suitable, trying test_loader_final.")
        loader_for_landscape = test_loader_final

    if not loader_for_landscape or len(loader_for_landscape.dataset) == 0:
        print("Error: No suitable data loader found for loss surface calculation.")
        return
    print(f"Using a dataloader with {len(loader_for_landscape.dataset)} samples for loss surface.")

    # 2. Define Model Creator Function
    num_classes = len(mynn_base_config.CLASSES)
    model_creator_fn = None
    if model_type == "DynamicCNN":
        if not architecture_config:
            print("Error: 'architecture_config' not found for DynamicCNN.")
            return

        copied_arch_config = copy.deepcopy(architecture_config)
        copied_exp_config = copy.deepcopy(exp_config)
        model_creator_fn = lambda: DynamicCNN(num_classes=num_classes,
                                              architecture_config=copied_arch_config,
                                              exp_config=copied_exp_config)
    else:
        print(f"Error: Model type '{model_type}' not supported for model_creator_fn.")
        return

    # 4. Get Criterion
    criterion = get_criterion(exp_config)

    # 5. Plot 3D Loss Surface
    print("\nStarting 3D loss surface generation...")
    for epoch in range(1, num_epochs + 1):
        # 每五轮训练的模型权重绘制一次3d损失图
        if epoch % 5 == 0:
            model_weights_path = os.path.join(experiment_root_path, f"epoch_{epoch}.pth")

            # Load Center Weights (Trained Model Weights)
            print(f"\nLoading center model weights from: {model_weights_path}")
            try:
                center_weights_state_dict = torch.load(
                    model_weights_path, map_location=torch.device('cpu')
                )
                print("Center model weights (state_dict) loaded successfully.")
            except FileNotFoundError:
                print(f"Error: Model weights file not found at {model_weights_path}")
                return
            except Exception as e:
                print(f"Error loading model weights state_dict: {e}")
                return

            plot_3d_loss_surface_plotly(
                model_creator_fn=model_creator_fn,
                center_weights_state_dict=center_weights_state_dict,
                dataloader=loader_for_landscape,
                criterion=criterion,
                device=device,
                output_dir=output_dir,
                exp_name=safe_exp_name,
                n_points=n_points,
                range_scale_alpha=range_scale_alpha,
                range_scale_beta=range_scale_beta,
                num_batches_for_loss_surface=num_batches_for_loss_surface,
                train_epoch=epoch,
            )

    # 绘制最优模型权重的3d损失图
    model_weights_path = os.path.join(experiment_root_path, f"model_best_dynamic_cnn_deeper.pth")
    print(f"\nLoading center model weights from: {model_weights_path}")
    try:
        center_weights_state_dict = torch.load(
            model_weights_path, map_location=torch.device('cpu')
        )
        print("Center model weights (state_dict) loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path}")
        return
    except Exception as e:
        print(f"Error loading model weights state_dict: {e}")
        return

    plot_3d_loss_surface_plotly(
        model_creator_fn=model_creator_fn,
        center_weights_state_dict=center_weights_state_dict,
        dataloader=loader_for_landscape,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        exp_name=safe_exp_name,
        n_points=30,
        range_scale_alpha=0.1,
        range_scale_beta=0.1,
        num_batches_for_loss_surface=10,
    )

    print("\n3D Loss surface visualization script finished.")


if __name__ == '__main__':
    experiment_time = '20250524-224302'
    experiment_name = 'dynamic_cnn_deeper'

    EXPERIMENT_ROOT_PATH = f"./runs/{experiment_time}/{experiment_name}"
    CONFIG_FILE_PATH = os.path.join(EXPERIMENT_ROOT_PATH, "full_experiment_config.json")

    OUTPUT_VIS_BASE_DIR = f"visualizations/3d loss surface/{experiment_time}"

    visualization_parameters = {
        "n_points": 15,
        "range_scale_alpha": 0.05,
        "range_scale_beta": 0.05,
        "num_batches_for_loss_surface": 10
    }

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Main config file {CONFIG_FILE_PATH} not found.")
    elif not os.path.exists(EXPERIMENT_ROOT_PATH):
        print(f"Error: Trained model weights {EXPERIMENT_ROOT_PATH} not found.")
    else:
        main_visualize_3d_loss(
            config_path=CONFIG_FILE_PATH,
            experiment_root_path=EXPERIMENT_ROOT_PATH,
            output_base_dir=OUTPUT_VIS_BASE_DIR,
            n_points=visualization_parameters['n_points'],
            range_scale_alpha=visualization_parameters['range_scale_alpha'],
            range_scale_beta=visualization_parameters['range_scale_beta'],
            num_batches_for_loss_surface=visualization_parameters['num_batches_for_loss_surface']
        )

