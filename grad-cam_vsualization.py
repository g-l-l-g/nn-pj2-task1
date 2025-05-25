# grad-cam_vsualization.py
import torch
import torch.nn as nn
import os
import json
import sys
import random


from mynn.models import DynamicCNN
from mynn.data_loader import get_cifar10_loaders
from mynn.utils.visualize_grad_cam import visualize_grad_cam
from mynn.utils.image_show import imshow
from mynn import config as mynn_base_config


def main_visualize_grad_cam(config_path, model_weights_path, output_base_dir="visualizations/grad_cam", num_samples=3):
    """
    Visualizes Grad-CAM for a few samples from the test set.
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

    safe_exp_name = exp_name.replace(' ', '_').replace('/', '_')
    output_dir = os.path.join(output_base_dir, safe_exp_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Grad-CAM visualizations will be saved to: {output_dir}")

    device = torch.device(mynn_base_config.DEVICE)
    print(f"Using device: {device}")

    # 1. Load Data (Test Loader)
    print("\nLoading test data...")
    _, _, test_loader, _ = get_cifar10_loaders(batch_size_override=max(4, num_samples), augment=False,
                                               val_split_ratio=None)
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

    # 4. Select Target Layer for Grad-CAM (e.g., last convolutional layer)
    target_layer_module = None
    if model_type == "DynamicCNN" and hasattr(model_instance, 'features_sequence'):
        for layer_module_gc in reversed(model_instance.features_sequence):
            if isinstance(layer_module_gc, nn.Conv2d):
                target_layer_module = layer_module_gc
                print(f"Using target layer for Grad-CAM: Last Conv2D in features_sequence")
                break

    if not target_layer_module:
        print("Error: Could not find a suitable Conv2D target layer for Grad-CAM.")
        # You might need to manually specify the target layer for other model types
        # e.g., target_layer_module = model_instance.layer4[-1].conv2 for a ResNet-like structure
        return

    # 5. Perform Grad-CAM on a few samples
    print(f"\nGenerating Grad-CAM for {num_samples} test samples...")
    samples_processed = 0
    for i, (batch_images_norm, batch_labels) in enumerate(test_loader):
        if samples_processed >= num_samples:
            break

        for j in range(batch_images_norm.size(0)):  # Iterate through images in the batch
            if samples_processed >= num_samples:
                break

            print(f"\nProcessing sample {samples_processed + 1}/{num_samples} (Batch {i + 1}, Item {j + 1})")

            # Prepare single image tensor for Grad-CAM (needs batch dimension)
            single_img_tensor_norm = batch_images_norm[j:j + 1].to(device)  # (1, C, H, W)
            true_label_idx = batch_labels[j].item()

            # Get model's prediction for this image
            with torch.no_grad():
                outputs = model_instance(single_img_tensor_norm)
                _, predicted_idx = torch.max(outputs, 1)
                predicted_idx = predicted_idx.item()

            # Show original image (optional, but helpful)
            orig_img_display = batch_images_norm[j].cpu()  # For imshow, (C,H,W) no batch dim
            sample_img_save_path = os.path.join(
                output_dir,
                f"sample_{samples_processed}_orig_true_{mynn_base_config.CLASSES[true_label_idx]}"
                f"_pred_{mynn_base_config.CLASSES[predicted_idx]}.png")
            imshow(
                orig_img_display,
                title=f"True: {mynn_base_config.CLASSES[true_label_idx]}, "
                      f"Pred: {mynn_base_config.CLASSES[predicted_idx]}",
                save_path=sample_img_save_path
            )

            # Generate and save Grad-CAM
            visualize_grad_cam(
                model=model_instance,
                target_layer=target_layer_module,
                input_tensor_normalized=single_img_tensor_norm,  # Already on device
                true_label_idx=true_label_idx,
                pred_label_idx=predicted_idx,
                class_names=mynn_base_config.CLASSES,
                output_dir=output_dir,
                file_name_prefix=f"gradcam_{safe_exp_name}_sample_{samples_processed}"
            )
            samples_processed += 1

    print("\nGrad-CAM visualization script finished.")


if __name__ == '__main__':
    experiment_time = '20250524-224302'
    experiment_name = 'dynamic_cnn_deeper'
    EXPERIMENT_ROOT_PATH = f"./runs/{experiment_time}/{experiment_name}"

    CONFIG_FILE_PATH = os.path.join(EXPERIMENT_ROOT_PATH, "full_experiment_config.json")
    TRAINED_MODEL_WEIGHTS_PATH = os.path.join(EXPERIMENT_ROOT_PATH, f"model_best_{experiment_name}.pth")
    OUTPUT_VIS_BASE_DIR = f"visualizations/grad_cam/{experiment_time}"

    NUM_SAMPLES_TO_VISUALIZE = 3

    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Main config file {CONFIG_FILE_PATH} not found.")
    elif not os.path.exists(TRAINED_MODEL_WEIGHTS_PATH):
        print(f"Error: Trained model weights {TRAINED_MODEL_WEIGHTS_PATH} not found.")
    else:
        main_visualize_grad_cam(
            config_path=CONFIG_FILE_PATH,
            model_weights_path=TRAINED_MODEL_WEIGHTS_PATH,
            output_base_dir=OUTPUT_VIS_BASE_DIR,
            num_samples=NUM_SAMPLES_TO_VISUALIZE
        )
