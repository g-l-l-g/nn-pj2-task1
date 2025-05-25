import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .. import config as mynn_config


def plot_confusion_matrix(all_preds, all_labels, class_names_list, exp_name="", output_dir=None):
    if not isinstance(all_preds, np.ndarray):
        all_preds = np.array(all_preds)
    if not isinstance(all_labels, np.ndarray):
        all_labels = np.array(all_labels)
    if all_preds.size == 0 or all_labels.size == 0:
        print("Warning: Predictions or labels are empty, cannot plot confusion matrix.")
        return

    max_label_val = len(class_names_list) - 1
    all_labels_clipped = np.clip(all_labels.astype(int), 0, max_label_val)
    all_preds_clipped = np.clip(all_preds.astype(int), 0, max_label_val)
    unique_labels_indices = np.arange(len(class_names_list))
    cm = confusion_matrix(all_labels_clipped, all_preds_clipped, labels=unique_labels_indices)

    plot_filename = f"confusion_matrix_{exp_name.replace(' ', '_')}.png"
    if not output_dir:
        output_dir = os.path.join(mynn_config.RUNS_DIR_BASE, exp_name.replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_filename)

    fig = plt.figure(
        figsize=(max(8, len(class_names_list) * 0.8), max(6, len(class_names_list) * 0.7)))  # Create figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list,
                annot_kws={"size": 8 if len(class_names_list) <= 10 else 6}, ax=fig.gca())  # Use gca()
    plt.xlabel('Predicted Label')  # Use plt for global labels if preferred, or ax.set_xlabel
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({exp_name})')
    fig.tight_layout()
    try:
        fig.savefig(plot_path)
        print(f"Confusion matrix saved to {plot_path}")
    except Exception as e:
        print(f"Error saving confusion matrix to {plot_path}: {e}")
    finally:
        plt.close(fig)
