import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from .. import config as mynn_config


def plot_training_results(train_losses, val_losses, train_accs, val_accs, exp_name="", output_dir=None):
    epochs_train = [i + 1 for i, (l, a) in enumerate(zip(train_losses, train_accs)) if not (np.isnan(l) or np.isnan(a))]
    train_losses_valid = [l for l in train_losses if not np.isnan(l)]
    train_accs_valid = [a for a in train_accs if not np.isnan(a)]

    epochs_val = [i + 1 for i, (l, a) in enumerate(zip(val_losses, val_accs)) if
                  not (np.isnan(l) or np.isnan(a) or l is None or a is None)]
    val_losses_valid = [l for l in val_losses if l is not None and not np.isnan(l)]
    val_accs_valid = [a for a in val_accs if a is not None and not np.isnan(a)]

    plot_filename = f"training_plot_{exp_name.replace(' ', '_')}.png"
    if not output_dir:
        output_dir = os.path.join(mynn_config.RUNS_DIR_BASE, exp_name.replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_filename)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    if train_losses_valid and epochs_train:
        ax1.plot(epochs_train, train_losses_valid, label='Training Loss')
    if val_losses_valid and epochs_val and any(not np.isnan(vl) for vl in val_losses_valid):
        ax1.plot(epochs_val, val_losses_valid, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend(loc='upper right')
    ax1.set_title(f'Loss ({exp_name})')

    if train_accs_valid and epochs_train:
        ax2.plot(epochs_train, train_accs_valid, label='Training Accuracy')
    if val_accs_valid and epochs_val and any(not np.isnan(va) for va in val_accs_valid):
        ax2.plot(epochs_val, val_accs_valid, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.set_title(f'Accuracy ({exp_name})')

    fig.tight_layout()
    try:
        fig.savefig(plot_path)  # Save the specific figure
        print(f"Training plot saved to {plot_path}")
    except Exception as e:
        print(f"Error: Could not save training plot to {plot_path}: {e}")
    finally:
        plt.close(fig)
