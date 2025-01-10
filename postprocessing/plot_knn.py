import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from PIL import Image
from sklearn.neighbors import NearestNeighbors

# Load dataset path from dataset_config.yaml
config_path = "configs/dataset_config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Ensure dataset path exists
if "dataset" in config and "data_path" in config["dataset"]:
    DATASET_PATH = config["dataset"]["data_path"]
else:
    raise KeyError("Error: 'dataset' or 'data_path' not found in dataset_config.yaml.")

def get_image_as_np_array(filename):
    """Returns an image as a numpy array from the dataset folder."""
    img_path = os.path.join(DATASET_PATH, filename)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    img = Image.open(img_path)
    return np.asarray(img)

def plot_knn_examples(embeddings, filenames, n_neighbors=3, num_examples=6):
    """Plots random images with their nearest neighbors."""
    
    if len(embeddings) != len(filenames):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs. {len(filenames)} filenames!")

    # Fit Nearest Neighbors Model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Select random images to visualize
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    for idx in samples_idx:
        fig = plt.figure(figsize=(15, 5))
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            ax = fig.add_subplot(1, n_neighbors, plot_x_offset + 1)
            fname = filenames[neighbor_idx]  # Correctly using dataset path
            img = get_image_as_np_array(fname)
            plt.imshow(img)
            ax.set_title(f"Dist: {distances[idx][plot_x_offset]:.3f}")
            plt.axis("off")
        plt.show()

if __name__ == "__main__":
    try:
        # Load embeddings and filenames
        embeddings = np.load("embeddings.npy")
        
        # Dynamically retrieve filenames
        filenames = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith((".jpg", ".png", ".jpeg"))])

        plot_knn_examples(embeddings, filenames)
    except FileNotFoundError:
        print("Error: Missing embeddings.npy. Run `main.sh` first.")
