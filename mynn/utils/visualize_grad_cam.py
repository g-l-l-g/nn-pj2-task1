import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
from captum.attr import LayerGradCam, visualization as viz

from .. import config as mynn_config


def visualize_grad_cam(model, target_layer, input_tensor_normalized, true_label_idx,
                       pred_label_idx, class_names, output_dir, file_name_prefix="grad_cam_viz"):
    model.eval()
    lgc = LayerGradCam(model, target_layer)

    device = next(model.parameters()).device
    input_tensor_normalized = input_tensor_normalized.to(device)

    attribution_pred = lgc.attribute(input_tensor_normalized, target=pred_label_idx)
    attribution_pred_cpu = attribution_pred.cpu()  # Move to CPU once for processing

    img_to_viz = input_tensor_normalized.squeeze(0).cpu() / 2 + 0.5
    img_to_viz = img_to_viz.clip(0, 1)

    attr_map_for_viz_pred = attribution_pred_cpu.squeeze(0).permute(1, 2, 0).detach().numpy()
    if np.abs(np.max(attr_map_for_viz_pred) - np.min(attr_map_for_viz_pred)) < 1e-5:
        attr_map_for_viz_pred = attr_map_for_viz_pred + np.random.normal(0, 1e-6, attr_map_for_viz_pred.shape)

    try:
        fig_pred, _ = viz.visualize_image_attr(
            attr=attr_map_for_viz_pred,  # MODIFIED HERE
            original_image=img_to_viz.permute(1, 2, 0).numpy(),
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title=f"Grad-CAM (Pred: {class_names[pred_label_idx]}, True: {class_names[true_label_idx]})",
            use_pyplot=False
        )
        grad_cam_save_path_pred = os.path.join(
            output_dir,
            f"{file_name_prefix}_pred_{class_names[pred_label_idx]}_true_{class_names[true_label_idx]}.png")
        fig_pred.savefig(grad_cam_save_path_pred)
        plt.close(fig_pred)
        print(f"Grad-CAM (predicted class) saved to {grad_cam_save_path_pred}")

    except AssertionError as e:
        print(f"Error during Grad-CAM visualization for predicted class {pred_label_idx} "
              f"(True: {true_label_idx}): {e}")
        print(f"Attribution stats: min={attribution_pred_cpu.min()}, "
              f"max={attribution_pred_cpu.max()}, mean={attribution_pred_cpu.mean()}")
    except Exception as e_gen:
        print(f"Generic error during Grad-CAM visualization for predicted class {pred_label_idx}: {e_gen}")

    if true_label_idx != pred_label_idx:
        attribution_true = lgc.attribute(input_tensor_normalized, target=true_label_idx)
        attribution_true_cpu = attribution_true.cpu()

        attr_map_for_viz_true = attribution_true_cpu.squeeze(0).permute(1, 2, 0).detach().numpy()
        if np.abs(np.max(attr_map_for_viz_true) - np.min(attr_map_for_viz_true)) < 1e-5:
            attr_map_for_viz_true = (
                    attr_map_for_viz_true + np.random.normal(0, 1e-6, attr_map_for_viz_true.shape)
            )

        try:
            fig_true, _ = viz.visualize_image_attr(
                attr=attr_map_for_viz_true,
                original_image=img_to_viz.permute(1, 2, 0).numpy(),
                method="blended_heat_map",
                sign="all",
                show_colorbar=True,
                title=f"Grad-CAM (Target: True Class {class_names[true_label_idx]})",
                use_pyplot=False
            )
            grad_cam_save_path_true = os.path.join(
                output_dir, f"{file_name_prefix}_target_true_{class_names[true_label_idx]}.png")
            fig_true.savefig(grad_cam_save_path_true)
            plt.close(fig_true)
        except AssertionError as e:
            print(f"Error during Grad-CAM visualization for true class {true_label_idx}: {e}")
            print(f"Attribution stats: min={attribution_true_cpu.min()}, "
                  f"max={attribution_true_cpu.max()}, mean={attribution_true_cpu.mean()}")
        except Exception as e_gen:
            print(f"Generic error during Grad-CAM visualization for true class {true_label_idx}: {e_gen}")