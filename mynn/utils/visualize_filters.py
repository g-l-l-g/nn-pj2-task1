import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

import seaborn as sns
import torch
import torch.nn as nn


def visualize_filters(layer_or_model, layer_identifier=None, exp_name="", output_dir=None):
    filters, target_layer = None, None
    layer_name_for_title = "conv_layer"
    if isinstance(layer_or_model, nn.Conv2d):
        target_layer = layer_or_model
        layer_name_for_title = f"{layer_or_model.__class__.__name__}_direct"
    elif isinstance(layer_or_model, nn.Module) and layer_identifier:
        try:
            attrs = layer_identifier.split('.')
            current_obj = layer_or_model
            for attr in attrs:
                current_obj = current_obj[int(attr)] if attr.isdigit() else getattr(current_obj, attr)
            if isinstance(current_obj, nn.Conv2d):
                target_layer = current_obj
                layer_name_for_title = layer_identifier.replace('.', '_')
            else:
                print(f"Identified layer '{layer_identifier}' is not a Conv2d layer, but {type(current_obj)}.")
                return
        except (AttributeError, IndexError, KeyError) as e:
            print(f"Could not access layer '{layer_identifier}' in the model: {e}")
            return
    else:
        print("Invalid input for visualize_filters. Provide a Conv2d layer or a model and layer_identifier.")
        return
    if target_layer is None:
        print("No target Conv2d layer found for visualization.")
        return
    filters = target_layer.weight.data.clone().cpu()
    if filters.numel() == 0:
        print("Filters are empty or have no elements.")
        return

    plot_filename = f"filters_{layer_name_for_title}_{exp_name.replace(' ', '_')}.png"
    if not output_dir:
        from . import config as mynn_config
        output_dir = os.path.join(mynn_config.RUNS_DIR_BASE, exp_name.replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, plot_filename)

    if filters.shape[1] > 3:
        filters_to_show = filters[:, :1, :, :]
    elif filters.shape[1] == 2:
        temp = torch.zeros(filters.shape[0], 3, filters.shape[2], filters.shape[3])
        temp[:, :2, :, :] = filters
        filters_to_show = temp
    elif filters.shape[1] == 1:
        filters_to_show = filters
    else:
        filters_to_show = filters

    num_filters = filters_to_show.shape[0]
    min_val, max_val = filters_to_show.min(), filters_to_show.max()
    filters_normalized = (filters_to_show - min_val) / (max_val - min_val + 1e-5)
    n_columns = min(8, num_filters)
    n_rows = (num_filters + n_columns - 1) // n_columns

    fig = plt.figure(figsize=(n_columns * 2, n_rows * 2))  # Create a new figure
    for i in range(num_filters):
        ax = fig.add_subplot(n_rows, n_columns, i + 1)
        if filters_normalized.shape[1] == 3:
            filt_img = filters_normalized[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(filt_img)
        elif filters_normalized.shape[1] == 1:
            filt_img = filters_normalized[i, 0].cpu().numpy()
            ax.imshow(filt_img, cmap='gray')
        else:
            ax.imshow(np.zeros_like(filters_normalized[i, 0].cpu().numpy()), cmap='gray')
        ax.axis('off')
        ax.set_title(f'F{i}')
    fig.suptitle(f'Filters ({exp_name} - {layer_name_for_title})', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        fig.savefig(plot_path)  # Save the specific figure
        print(f"Filter visualization saved to {plot_path}")
    except Exception as e:
        print(f"Error saving filter plot to {plot_path}: {e}")
    finally:
        plt.close(fig)
