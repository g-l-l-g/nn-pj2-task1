# confusion_matrix_visualization.py
import torch
import os
import json
import sys
import numpy as np
from tqdm import tqdm

from mynn.models import DynamicCNN
from mynn.data_loader import get_cifar10_loaders
from mynn.utils.plot_confusion_matrix import plot_confusion_matrix
from mynn import config as mynn_base_config


def main_visualize_confusion_matrix(config_path, model_weights_path, output_base_dir="visualizations/confusion_matrix"):
    """
    Generates and saves a confusion matrix for the model on the test set.
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

    model_type = exp_config.get("model_type")
    architecture_config = exp_config.get("architecture_config")
    exp_name = exp_config.get("architecture_name", "unknown_experiment")
    batch_size = exp_config.get("batch_size", 32)  # Use batch size from config for eval

    safe_exp_name = exp_name.replace(' ', '_').replace('/', '_')
    output_dir = os.path.join(output_base_dir, safe_exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Confusion matrix will be saved to: {output_dir}")

    device = torch.device(mynn_base_config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data (Test Loader)
    print("\nLoading test data for confusion matrix...")
    _, _, test_loader, _ = get_cifar10_loaders(batch_size_override=batch_size, augment=False, val_split_ratio=None)
    if not test_loader:
        print("Error: Failed to load test data.")
        return

    # 2. Initialize Model
    print("\nInitializing model...")
    num_classes = len(mynn_base_config.CLASSES)
    model_instance = None
    if model_type == "DynamicCNN":
        if not architecture_config:
            print("Error: 'architecture_config' not found for DynamicCNN.")
            return
        model_instance = DynamicCNN(num_classes=num_classes,
                                    architecture_config=architecture_config,
                                    exp_config=exp_config)
    else:
        print(f"Error: Model type '{model_type}' not supported.")
        return

    # 3. Load Trained Weights
    print(f"\nLoading trained model weights from: {model_weights_path}")
    try:
        model_instance.load_state_dict(torch.load(model_weights_path, map_location=device))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model_instance.to(device)
    model_instance.eval()

    # 4. Get Predictions on Test Set
    print("\nGetting predictions from test set...")
    all_preds_list = []
    all_labels_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_instance(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds_list.extend(predicted.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    if not all_preds_list or not all_labels_list:
        print("No predictions or labels collected. Cannot generate confusion matrix.")
        return

    # 5. Plot Confusion Matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(
        all_preds=np.array(all_preds_list),
        all_labels=np.array(all_labels_list),
        class_names_list=mynn_base_config.CLASSES,  # Pass the list of class names
        exp_name=safe_exp_name,
        output_dir=output_dir
    )

    print("\nConfusion matrix visualization script finished.")


if __name__ == '__main__':

    experiment_time = '20250524-224302'
    experiment_name = 'dynamic_cnn_deeper'
    EXPERIMENT_ROOT_PATH = f"./runs/{experiment_time}/{experiment_name}"

    CONFIG_FILE_PATH = os.path.join(EXPERIMENT_ROOT_PATH, "full_experiment_config.json")
    TRAINED_MODEL_WEIGHTS_PATH = os.path.join(EXPERIMENT_ROOT_PATH, f"model_best_{experiment_name}.pth")
    OUTPUT_VIS_BASE_DIR = f"visualizations/confusion_matrix/{experiment_time}"

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Main config file {CONFIG_FILE_PATH} not found.")
    elif not os.path.exists(TRAINED_MODEL_WEIGHTS_PATH):
        print(f"Error: Trained model weights {TRAINED_MODEL_WEIGHTS_PATH} not found.")
    else:
        main_visualize_confusion_matrix(
            config_path=CONFIG_FILE_PATH,
            model_weights_path=TRAINED_MODEL_WEIGHTS_PATH,
            output_base_dir=OUTPUT_VIS_BASE_DIR
        )
