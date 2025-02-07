import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import csv

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

def plot_knn_for_specific_image(embeddings, filenames, query_filename, n_neighbors=3):
    """
    Plots the nearest neighbors of a specific image.

    Args:
        embeddings (np.ndarray): Embeddings array.
        filenames (list): List of filenames corresponding to embeddings.
        query_filename (str): Filename of the query image to find neighbors for.
        n_neighbors (int): Number of nearest neighbors to retrieve.
    """
    if len(embeddings) != len(filenames):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs. {len(filenames)} filenames!")

    # Ensure the query file exists
    query_filename_normalized = os.path.splitext(os.path.basename(query_filename))[0]
    normalized_filenames = [os.path.splitext(os.path.basename(fname))[0] for fname in filenames]
    if query_filename_normalized not in normalized_filenames:
        raise FileNotFoundError(f"Query file '{query_filename}' not found in the dataset.")

    # Find the index of the query file
    query_index = normalized_filenames.index(query_filename_normalized)

    # Fit Nearest Neighbors Model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Plot the query image and its neighbors
    fig = plt.figure(figsize=(15, 5))
    for plot_x_offset, neighbor_idx in enumerate(indices[query_index]):
        ax = fig.add_subplot(1, n_neighbors, plot_x_offset + 1)
        fname = filenames[neighbor_idx]  # Correctly using dataset path
        img = get_image_as_np_array(fname)
        plt.imshow(img)
        if neighbor_idx == query_index:
            ax.set_title("Query Image")
        else:
            ax.set_title(f"Dist: {distances[query_index][plot_x_offset]:.3f}")
        plt.axis("off")
    plt.show()


def load_labels_from_csv(csv_path, label_column):
    """
    Load labels from a CSV file.

    Args:
        csv_path (str): Path to the CSV file.
        label_column (str): The column name to use as labels.

    Returns:
        dict: A mapping of filename (without extension) to label value.
    """
    labels = {}
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = os.path.splitext(row["Filename"])[0]
            labels[filename] = row[label_column]
    return labels

def write_knn_labels_to_file(embeddings, filenames, query_filename, csv_path, label_column, output_path, n_neighbors=3):
    """
    Write the labels of the nearest neighbors for a specific image to a file.

    Args:
        embeddings (np.ndarray): Embeddings array.
        filenames (list): List of filenames corresponding to embeddings.
        query_filename (str): Filename of the query image to find neighbors for.
        csv_path (str): Path to the CSV file containing labels.
        label_column (str): The column name in the CSV to use as labels.
        output_path (str): Path to save the output file with labels.
        n_neighbors (int): Number of nearest neighbors to retrieve.
    """
    if len(embeddings) != len(filenames):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs. {len(filenames)} filenames!")

    # Load labels
    labels = load_labels_from_csv(csv_path, label_column)

    # Ensure the query file exists
    query_filename_normalized = os.path.splitext(os.path.basename(query_filename))[0]
    normalized_filenames = [os.path.splitext(os.path.basename(fname))[0] for fname in filenames]
    if query_filename_normalized not in normalized_filenames:
        raise FileNotFoundError(f"Query file '{query_filename}' not found in the dataset.")

    # Find the index of the query file
    query_index = normalized_filenames.index(query_filename_normalized)

    # Fit Nearest Neighbors Model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Write labels to file
    with open(output_path, "w") as output_file:
        output_file.write("Neighbor Index,Filename,Distance,Label\n")
        for neighbor_idx in indices[query_index]:
            fname = os.path.splitext(os.path.basename(filenames[neighbor_idx]))[0]
            distance = distances[query_index][list(indices[query_index]).index(neighbor_idx)]
            label = labels.get(fname, "N/A")
            output_file.write(f"{neighbor_idx},{fname},{distance:.3f},{label}\n")
    
    print(f"Nearest neighbor labels written to: {output_path}")

def plot_knn_with_labels(embeddings, filenames, query_filename, csv_path, label_column, n_neighbors=3):
    """
    Plots the nearest neighbors of a specific image with labels on the plot.

    Args:
        embeddings (np.ndarray): Embeddings array.
        filenames (list): List of filenames corresponding to embeddings.
        query_filename (str): Filename of the query image to find neighbors for.
        csv_path (str): Path to the CSV file containing labels.
        label_column (str): The column name in the CSV to use as labels.
        n_neighbors (int): Number of nearest neighbors to retrieve.
    """
    if len(embeddings) != len(filenames):
        raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs. {len(filenames)} filenames!")

    # Load labels
    labels = load_labels_from_csv(csv_path, label_column)

    # Ensure the query file exists
    query_filename_normalized = os.path.splitext(os.path.basename(query_filename))[0]
    normalized_filenames = [os.path.splitext(os.path.basename(fname))[0] for fname in filenames]
    if query_filename_normalized not in normalized_filenames:
        print(f"Warning: Query file '{query_filename}' not found in the dataset. Skipping.")
        return  # Skip this query

    # Find the index of the query file
    query_index = normalized_filenames.index(query_filename_normalized)

    # Fit Nearest Neighbors Model
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Plot the query image and its neighbors
    fig = plt.figure(figsize=(15, 5))
    for plot_x_offset, neighbor_idx in enumerate(indices[query_index]):
        ax = fig.add_subplot(1, n_neighbors, plot_x_offset + 1)
        fname = filenames[neighbor_idx]  # Correctly using dataset path
        img = get_image_as_np_array(fname)
        plt.imshow(img)
        if neighbor_idx == query_index:
            ax.set_title("Query Image")
        else:
            label_value = labels.get(os.path.splitext(os.path.basename(fname))[0], "N/A")
            ax.set_title(f"Label: {label_value}")
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    try:
        # Load embeddings and filenames
        embeddings = np.load("embeddings.npy")
        
        # Dynamically retrieve filenames
        filenames = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith((".jpg", ".png", ".jpeg"))])

        plot_knn_examples(embeddings, filenames)


        query_filename = "example_image.jpg" 
        csv_path = "path/to/your/csv_file.csv"
        label_column = "YourLabelColumn"  # Replace with the label column name
        output_path = "nearest_neighbors_labels.csv" # Replace with your query filename
        plot_knn_for_specific_image(embeddings, filenames, query_filename)

        write_knn_labels_to_file(
            embeddings, 
            filenames, 
            query_filename, 
            csv_path, 
            label_column, 
            output_path
        )

        plot_knn_with_labels(
            embeddings, 
            filenames, 
            query_filename, 
            csv_path, 
            label_column, 
            n_neighbors=5
        )


    except FileNotFoundError:
        print("Error: Missing embeddings.npy or query image. Run `main.sh` first.")
