import numpy as np
import matplotlib.pyplot as plt

def plot_umap_projection(embeddings_2d):
    """Plots the UMAP embeddings."""
    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=5, alpha=0.7)
    plt.show()
