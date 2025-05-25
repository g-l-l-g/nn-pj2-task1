import torch
import os
import json

from mynn.models import DynamicCNN
from mynn.data_loader import get_cifar10_loaders
from mynn.evaluate import evaluate_model
from mynn.criterion import get_criterion
from mynn import config as mynn_base_config


def test_model_on_test_set(config_path_, model_weights_path_):
    """
    Loads a trained model based on configuration and weights,
    then evaluates it on the CIFAR-10 test set and prints the accuracy.
    """
    # 1. Load Experiment Configuration
    print(f"Loading configuration from: {config_path_}")
    try:
        with open(config_path_, 'r') as f:
            exp_config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path_}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {config_path_}")
        return

    model_type = exp_config.get("model_type")
    architecture_config = exp_config.get("architecture_config")
    exp_name_from_config = exp_config.get("architecture_name", "unknown_model")
    batch_size = exp_config.get("batch_size", mynn_base_config.DEFAULT_BATCH_SIZE)

    print(f"\n--- Testing Model: {exp_name_from_config} ---")
    print(f"  Model Type: {model_type}")
    print(f"  Config File: {config_path_}")
    print(f"  Weights File: {model_weights_path_}")

    # 2. Determine Device
    device = torch.device(mynn_base_config.DEVICE)
    print(f"Using device: {device}")

    # 3. Load Test Data
    print("\nLoading CIFAR-10 test data...")

    try:
        _, _, test_loader, _ = get_cifar10_loaders(
            batch_size_override=batch_size,
            augment=False,
            val_split_ratio=None,
            num_workers=2
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not test_loader:
        print("Error: Failed to load test data.")
        return
    print(f"Test data loaded: {len(test_loader.dataset)} samples in {len(test_loader)} batches.")

    # 4. Initialize Model
    print("\nInitializing model...")
    num_classes = len(mynn_base_config.CLASSES)
    model_instance = None

    if model_type == "DynamicCNN":
        if not architecture_config:
            print("Error: 'architecture_config' not found in experiment config for DynamicCNN.")
            return
        model_instance = DynamicCNN(num_classes=num_classes,
                                    architecture_config=architecture_config,
                                    exp_config=exp_config)

    else:
        print(f"Error: Model type '{model_type}' not currently supported by this test script.")
        print("Please extend this script or ensure your model can be loaded via DynamicCNN config.")
        return

    # 5. Load Trained Weights
    print(f"\nLoading trained model weights from: {model_weights_path_}")
    try:
        # Load weights to CPU first to avoid GPU memory issues if the saved model was large
        # and the current machine has less GPU RAM. Then move model to device.
        state_dict = torch.load(model_weights_path_, map_location=torch.device('cpu'))
        model_instance.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {model_weights_path_}")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model_instance.to(device)
    model_instance.eval()

    # 6. Define Criterion for Evaluation (Optional if evaluate_model can work without it for accuracy)
    try:
        criterion_eval = get_criterion(exp_config)
    except ValueError as e:
        print(f"Warning: Could not get criterion from config: {e}. Proceeding without loss calculation if possible.")
        criterion_eval = None

    # 7. Evaluate on Test Set
    print("\nEvaluating model on the test set...")
    try:
        test_loss, test_acc, test_error, _ = evaluate_model(
            model=model_instance,
            testloader=test_loader,
            criterion=criterion_eval,
            device_override=device,
            exp_name=f"{exp_name_from_config}_test_run",
            output_dir=None,
            num_classes=num_classes
        )
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    # 8. Print Results
    print("\n--- Test Results ---")
    print(f"Model: {exp_name_from_config}")
    print(f"Configuration: {config_path_}")
    print(f"Weights: {model_weights_path_}")
    if test_loss is not None:
        print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    if test_error is not None:
        print(f"  Test Error: {test_error:.4f}")

    print("\nTesting script finished.")


if __name__ == '__main__':

    ROOT_DIR = "./runs/20250524-224302/dynamic_cnn_deeper"
    config = "full_experiment_config.json"
    weights = "model_best_dynamic_cnn_deeper.pth"
    # weights = "epoch_40.pth"
    config_path = os.path.join(ROOT_DIR, config)
    model_weights_path = os.path.join(ROOT_DIR, weights)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config}' not found.")
        raise FileNotFoundError
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file '{weights}' not found.")
        raise FileNotFoundError
    test_model_on_test_set(config_path_=config_path, model_weights_path_=model_weights_path)
