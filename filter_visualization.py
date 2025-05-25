# filter_visualization.py
import torch
import torch.nn as nn
import os
import json

from mynn.models import DynamicCNN
from mynn.utils.visualize_filters import visualize_filters
from mynn import config as mynn_base_config


def main_visualize_filters(config_path, model_weights_path, output_base_dir="visualizations/filters"):
    """
    Visualizes filters of a trained model.
    """

    # 1. Load Experiment Configuration
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
    architecture_config = exp_config.get("architecture_config")  # Specific to DynamicCNN
    exp_name = exp_config.get("architecture_name", "unknown_experiment")  # Use a descriptive name

    safe_exp_name = exp_name.replace(' ', '_').replace('/', '_')
    output_dir = os.path.join(output_base_dir, safe_exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Filter visualizations will be saved to: {output_dir}")

    # 2. Determine Device
    device = torch.device(mynn_base_config.DEVICE)  # Use device from base config
    print(f"Using device: {device}")

    # 3. Initialize Model
    print("\nInitializing model...")
    num_classes = len(mynn_base_config.CLASSES)  # Assuming CIFAR-10
    model_instance = None

    if model_type == "DynamicCNN":
        if not architecture_config:
            print("Error: 'architecture_config' not found in experiment config for DynamicCNN.")
            return
        model_instance = DynamicCNN(num_classes=num_classes,
                                    architecture_config=architecture_config,
                                    exp_config=exp_config)  # Pass full exp_config
    else:
        print(f"Error: Model type '{model_type}' not supported by this script.")
        return

    # 4. Load Trained Weights
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

    # 5. Visualize Filters (e.g., first convolutional layer)
    print("\nVisualizing filters...")
    first_conv_identifier = None
    if model_type == "DynamicCNN" and hasattr(model_instance, 'features_sequence'):
        for i, layer_mod in enumerate(model_instance.features_sequence):
            if isinstance(layer_mod, nn.Conv2d):
                # Construct the identifier string to access this layer
                # This assumes 'features_sequence' is an nn.Sequential or nn.ModuleList
                first_conv_identifier = f'features_sequence.{i}'
                break

    if first_conv_identifier:
        print(f"Attempting to visualize filters for layer: {first_conv_identifier}")
        visualize_filters(model_instance,
                          layer_identifier=first_conv_identifier,
                          exp_name=safe_exp_name,
                          output_dir=output_dir)
    else:
        print("Could not automatically find the first Conv2D layer for DynamicCNN.")
        # You might need to manually specify the layer_identifier if the model structure is different
        # e.g., if model_instance.conv1 exists: visualize_filters(model_instance.conv1, ...)

    print("\nFilter visualization script finished.")


if __name__ == '__main__':
    # --- Configuration for the script ---
    experiment_time = '20250524-224302'
    experiment_name = 'dynamic_cnn_deeper'
    EXPERIMENT_ROOT_PATH = f"./runs/{experiment_time}/{experiment_name}"

    CONFIG_FILE_PATH = os.path.join(EXPERIMENT_ROOT_PATH, "full_experiment_config.json")
    TRAINED_MODEL_WEIGHTS_PATH = os.path.join(EXPERIMENT_ROOT_PATH, f"model_best_{experiment_name}.pth")
    OUTPUT_VIS_BASE_DIR = f"visualizations/filters/{experiment_time}"

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Main config file {CONFIG_FILE_PATH} not found. Please create it or specify the correct path.")
    elif not os.path.exists(TRAINED_MODEL_WEIGHTS_PATH):
        print(
            f"Error: Trained model weights {TRAINED_MODEL_WEIGHTS_PATH} not found."
            f" Please train a model first or specify the correct path.")
    else:
        main_visualize_filters(
            config_path=CONFIG_FILE_PATH,
            model_weights_path=TRAINED_MODEL_WEIGHTS_PATH,
            output_base_dir=OUTPUT_VIS_BASE_DIR
        )