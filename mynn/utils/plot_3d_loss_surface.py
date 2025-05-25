# plot_3d_loss_surface.py (REVISED)

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import os
from tqdm import tqdm
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def _generate_random_direction(trainable_float_state_dict, device):  # Changed input name for clarity
    """
    Generates a random direction vector.
    Input: trainable_float_state_dict - A state_dict containing ONLY the tensors of
                                        parameters that are floating_point and require_grad.
    """
    direction = type(trainable_float_state_dict)()
    for k, v_param_tensor in trainable_float_state_dict.items():
        # Assumption: All v_param_tensor here are float and from trainable params.
        # The requires_grad check is implicitly handled by how trainable_float_state_dict is constructed.
        random_v = torch.randn_like(v_param_tensor, device=device)
        norm_v_base = torch.norm(v_param_tensor.float())  # v_param_tensor is already the tensor data
        norm_random_v = torch.norm(random_v.float())

        if norm_random_v > 1e-10:
            random_v_normalized = random_v * (norm_v_base / norm_random_v)
        else:
            random_v_normalized = random_v
        direction[k] = random_v_normalized
    return direction


def _calculate_loss_at_point(model, dataloader, criterion, device, num_batches_for_loss_surface=5):
    # ... (implementation is likely okay, ensure model.eval() is called) ...
    model.eval()
    running_loss = 0.0
    total_samples = 0
    if len(dataloader) == 0:
        return float('nan')
    num_batches_to_eval = min(num_batches_for_loss_surface, len(dataloader))
    if num_batches_to_eval == 0 and len(dataloader) > 0 and num_batches_for_loss_surface > 0:
        num_batches_to_eval = 1
    elif num_batches_to_eval == 0 and num_batches_for_loss_surface == 0 and len(dataloader) > 0:
        return float('nan')

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if i >= num_batches_to_eval:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            running_loss += loss_val.item() * inputs.size(0)
            total_samples += inputs.size(0)
    if total_samples == 0:
        return float('nan')
    return running_loss / total_samples


def plot_3d_loss_surface_plotly(
        model_creator_fn,
        center_weights_state_dict,
        dataloader,
        criterion,
        device,
        output_dir,
        exp_name,
        n_points=7,
        range_scale_alpha=0.1,
        range_scale_beta=0.1,
        num_batches_for_loss_surface=5,
        train_epoch=None,
):
    print(f"\n--- Generating INTERACTIVE 3D Loss Surface for {exp_name} (Grid: {n_points}x{n_points}) ---")
    if dataloader is None or len(dataloader.dataset) == 0:
        print("Dataloader for loss surface is empty. Skipping.")
        return
    if n_points < 2:
        print("n_points must be at least 2. Skipping.")
        return

    temp_model_for_structure = model_creator_fn()  # Used to identify trainable params by name
    temp_model_for_structure.to(device)

    trainable_float_param_names = {
        name for name, param in temp_model_for_structure.named_parameters()
        if param.is_floating_point() and param.requires_grad
    }

    if not trainable_float_param_names:
        print("Error: Model structure (from model_creator_fn) has no trainable floating point parameters.")
        print("Detailed parameter check of fresh model instance:")
        for name, param_v in temp_model_for_structure.named_parameters():
            v = param_v
            is_float = v.is_floating_point()
            req_grad = v.requires_grad
            print(f"  Param: {name}, IsFloat: {is_float}, ReqGrad: {req_grad}, Dtype: {v.dtype}, Shape: {v.shape}")
        return

    base_trainable_sd_values = {
        name: center_weights_state_dict[name].clone().to(device)
        for name in trainable_float_param_names
        if name in center_weights_state_dict  # Ensure the key exists in the loaded weights
    }

    if not base_trainable_sd_values:
        print(
            f"Error: No trainable float parameters from model structure found in center_weights_state_dict."
            f" Names were: {trainable_float_param_names}")
        return

    print("Generating random direction 1 (filter-wise normalized for trainable params)...")
    direction1_sd_trainable = _generate_random_direction(base_trainable_sd_values, device)
    print("Generating random direction 2 (filter-wise normalized for trainable params)...")
    direction2_sd_trainable = _generate_random_direction(base_trainable_sd_values, device)

    #  operates on the directions for trainable params
    dot_product_d1_d2 = 0.0
    norm_sq_d1 = 0.0
    for key in direction1_sd_trainable:
        dot_product_d1_d2 += torch.sum(direction1_sd_trainable[key] * direction2_sd_trainable[key])
        norm_sq_d1 += torch.sum(direction1_sd_trainable[key] * direction1_sd_trainable[key])
    if norm_sq_d1 > 1e-10:
        projection_factor = dot_product_d1_d2 / norm_sq_d1
        for key in direction2_sd_trainable:
            direction2_sd_trainable[key] -= projection_factor * direction1_sd_trainable[key]

    alpha_coords = np.linspace(-range_scale_alpha, range_scale_alpha, n_points)
    beta_coords = np.linspace(-range_scale_beta, range_scale_beta, n_points)
    Alpha_grid, Beta_grid = np.meshgrid(alpha_coords, beta_coords)
    losses_surface = np.full_like(Alpha_grid, float('nan'), dtype=float)

    # Model instance for calculations - create it once and load weights into it
    calc_model = model_creator_fn()
    calc_model.to(device)

    print(f"Calculating loss surface over a {n_points}x{n_points} grid...")
    with tqdm(total=n_points * n_points, desc="Loss Surface Calculation") as pbar:
        for i_idx in range(n_points):
            for j_idx in range(n_points):
                alpha = Alpha_grid[j_idx, i_idx]
                beta = Beta_grid[j_idx, i_idx]

                # Start with a full copy of the original center weights
                current_interpolated_full_sd = {k: v.clone() for k, v in center_weights_state_dict.items()}

                try:
                    # Apply perturbations ONLY to the parameters identified as trainable and float
                    for key in base_trainable_sd_values.keys():  # Iterate over names of trainable float params
                        w_center_val = base_trainable_sd_values[key]  # This is already on device
                        d1_val = direction1_sd_trainable[key]  # Already on device
                        d2_val = direction2_sd_trainable[key]  # Already on device

                        current_interpolated_full_sd[key] = w_center_val + alpha * d1_val + beta * d2_val

                    calc_model.load_state_dict(current_interpolated_full_sd, strict=False)

                    loss = _calculate_loss_at_point(calc_model, dataloader, criterion, device,
                                                    num_batches_for_loss_surface)
                    losses_surface[j_idx, i_idx] = loss
                except Exception as e:
                    losses_surface[j_idx, i_idx] = float('nan')

                pbar.set_postfix(
                    {"alpha": f"{alpha:.2f}", "beta": f"{beta:.2f}", "loss": f"{losses_surface[j_idx, i_idx]:.4f}"})
                pbar.update(1)

    # ... (Plotly plotting code remains largely the same) ...
    # Make sure to calculate center_loss_val using calc_model and center_weights_state_dict
    print("Plotting INTERACTIVE 3D loss surface with Plotly...")
    calc_model.load_state_dict(center_weights_state_dict, strict=False)
    center_loss_val = _calculate_loss_at_point(calc_model, dataloader, criterion, device, num_batches_for_loss_surface)

    fig_plotly = go.Figure(data=[go.Surface(
        z=losses_surface, x=Alpha_grid, y=Beta_grid, colorscale='Viridis',
        colorbar=dict(title='Loss'),
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)),
        hoverinfo='x+y+z', name='Loss Surface')])
    if not np.isnan(center_loss_val):
        fig_plotly.add_trace(go.Scatter3d(x=[0], y=[0], z=[center_loss_val], mode='markers',
                                          marker=dict(size=8, color='red', symbol='diamond'),
                                          name=f'Center Loss: {center_loss_val:.4f}'))
    min_loss_val_surface = np.nanmin(losses_surface)
    min_alpha_surface, min_beta_surface = None, None
    if not np.isnan(min_loss_val_surface):
        min_idx = np.unravel_index(np.nanargmin(losses_surface), losses_surface.shape)
        min_alpha_surface, min_beta_surface = Alpha_grid[min_idx], Beta_grid[min_idx]
        fig_plotly.add_trace(go.Scatter3d(x=[min_alpha_surface], y=[min_beta_surface], z=[min_loss_val_surface],
                                          mode='markers', marker=dict(size=8, color='cyan', symbol='circle'),
                                          name=f'Surface Min Loss: {min_loss_val_surface:.4f}'))
    fig_plotly.update_layout(
        title=f'Interactive 3D Loss Surface ({exp_name}, {n_points}x{n_points} grid)',
        scene=dict(xaxis_title=f'Alpha (Dir 1, Scale: {range_scale_alpha})',
                   yaxis_title=f'Beta (Dir 2, Scale: {range_scale_beta})',
                   zaxis_title='Loss', camera_eye=dict(x=1.8, y=1.8, z=1.4)),
        autosize=True, margin=dict(l=50, r=50, b=50, t=100))

    if train_epoch is not None:
        plot_filename_html = f"{exp_name.replace(' ', '_')}_epoch_{train_epoch}.html"
    else:
        plot_filename_html = f"{exp_name.replace(' ', '_')}.html"
    plot_filename_html = plot_filename_html.replace('.html', f'_{n_points}x{n_points}.html')

    os.makedirs(output_dir, exist_ok=True)
    plot_path_html = os.path.join(output_dir, plot_filename_html)
    try:
        fig_plotly.write_html(plot_path_html, include_plotlyjs='cdn')
        print(f"Interactive 3D Loss surface plot saved to {plot_path_html}")
    except Exception as e:
        print(f"Error saving interactive 3D loss surface plot: {e}")

    # Static Matplotlib Fallback (ensure variables like min_alpha_surface are defined)
    fig_static = None  # Initialize to prevent UnboundLocalError in finally
    try:
        print("Generating static Matplotlib 3D Loss Surface as fallback...")
        fig_static = plt.figure(figsize=(12, 8))
        ax_static = fig_static.add_subplot(111, projection='3d')
        Z_masked = np.ma.masked_invalid(losses_surface)
        surf_static = ax_static.plot_surface(Alpha_grid, Beta_grid, Z_masked, cmap='viridis', edgecolor='none',
                                             rstride=1, cstride=1)
        ax_static.set_xlabel(f'Alpha (Scale: {range_scale_alpha})')
        ax_static.set_ylabel(f'Beta (Scale: {range_scale_beta})')
        ax_static.set_zlabel('Loss')
        ax_static.set_title(f'Static 3D Loss Surface ({exp_name})')
        if not np.all(np.isnan(losses_surface)):
            fig_static.colorbar(surf_static, shrink=0.5, aspect=10, label='Loss')

        has_legend_items = False
        if not np.isnan(center_loss_val):
            ax_static.scatter([0], [0], [center_loss_val], color='red', s=60, edgecolor='black', depthshade=False,
                              label=f'Center: {center_loss_val:.2f}')
            has_legend_items = True
        if not np.isnan(min_loss_val_surface):
            ax_static.scatter([min_alpha_surface], [min_beta_surface], [min_loss_val_surface], color='cyan', s=60,
                              edgecolor='black', depthshade=False, label=f'Min on Surf: {min_loss_val_surface:.2f}')
            has_legend_items = True
        if has_legend_items:
            ax_static.legend()

        if train_epoch is not None:
            plot_filename_static = f"loss_landscape_3d_static_{exp_name.replace(' ', '_')}_epoch_{train_epoch}.png"
        else:
            plot_filename_static = f"loss_landscape_3d_static_{exp_name.replace(' ', '_')}.png"
        plot_filename_static = plot_filename_static.replace('.png', f'_{n_points}x{n_points}.png')

        plot_path_static = os.path.join(output_dir, plot_filename_static)
        fig_static.savefig(plot_path_static, dpi=150)
        print(f"Static 3D Loss surface plot saved to {plot_path_static}")
    except Exception as e:
        print(f"Error saving static 3D loss surface plot: {e}")
    finally:
        if fig_static:
            plt.close(fig_static)
