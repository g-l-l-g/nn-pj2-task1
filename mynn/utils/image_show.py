import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def imshow(img_tensor, title=None, save_path=None, unnormalize=True):
    if unnormalize:
        img_tensor = img_tensor / 2 + 0.5
    npimg = img_tensor.clip(0, 1).cpu().numpy()

    fig = plt.figure(figsize=(5, 5))  # Create a new figure
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)

    if save_path:
        try:
            fig.savefig(save_path)  # Save the specific figure
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Error saving image to {save_path}: {e}")
        finally:
            plt.close(fig)  # Close the specific figure
    else:
        plt.close(fig)
