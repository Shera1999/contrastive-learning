import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def normalize_to_grid(embeddings_2d, grid_size):
    """Normalize UMAP coordinates to a fixed grid size."""
    min_x, min_y = embeddings_2d.min(axis=0)
    max_x, max_y = embeddings_2d.max(axis=0)

    normalized_x = (embeddings_2d[:, 0] - min_x) / (max_x - min_x) * (grid_size - 1)
    normalized_y = (embeddings_2d[:, 1] - min_y) / (max_y - min_y) * (grid_size - 1)

    return np.clip(np.round(normalized_x).astype(int), 0, grid_size - 1), np.clip(np.round(normalized_y).astype(int), 0, grid_size - 1)

def plot_images_on_grid(embeddings_2d, filenames, grid_size=10, cell_size=128, step=20):
    """Plots images in a grid structure."""
    grid_canvas = np.zeros((grid_size * cell_size, grid_size * cell_size, 3), dtype=np.uint8)
    grid_x, grid_y = normalize_to_grid(embeddings_2d[::step], grid_size)

    for i, (x, y) in enumerate(zip(grid_x, grid_y)):
        try:
            img_path = filenames[i * step]
            with Image.open(img_path) as img:
                img = img.resize((cell_size, cell_size))
                img_np = np.array(img.convert("RGB"))
                x_start, y_start = y * cell_size, x * cell_size
                grid_canvas[x_start:x_start + cell_size, y_start:y_start + cell_size] = img_np
        except Exception as e:
            print(f"Error processing {filenames[i * step]}: {e}")

    plt.figure(figsize=(15, 15))
    plt.imshow(grid_canvas)
    plt.axis("off")
    plt.show()
