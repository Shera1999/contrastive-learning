import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os


def load_labels_from_csv(csv_path, filenames, label_column):
    """
    Loads labels from a CSV file and matches them with the provided filenames.

    Parameters:
        csv_path (str): Path to the CSV file containing filenames and labels.
        filenames (list): List of filenames to filter the labels.
        label_column (str): Column name in the CSV to use as the label.

    Returns:
        np.ndarray: Array of labels corresponding to the filenames.
    """
    # Load CSV
    data = pd.read_csv(csv_path)
    
    # Ensure the CSV contains the filename and label_column
    if "Filename" not in data.columns or label_column not in data.columns:
        raise ValueError(f"CSV file must contain 'Filename' and '{label_column}' columns.")

    # Filter based on filenames
    data_filtered = data[data["Filename"].isin(filenames)]
    if len(data_filtered) != len(filenames):
        print(f"Warning: {len(filenames) - len(data_filtered)} filenames do not have matching labels in the CSV.")

    # Ensure labels are aligned with the order of filenames
    data_filtered = data_filtered.set_index("Filename").reindex(filenames)
    labels = data_filtered[label_column].values

    return labels


def plot_hexbin_with_labels(
    embeddings_2d,
    filenames,
    csv_path,
    label_column,
    gridsize=30,
    save_path="hexbin_with_labels.png",
):
    """
    Plots a 2D hexbin histogram of embeddings with labels from a CSV.

    Args:
        embeddings_2d (np.ndarray): 2D array of UMAP embeddings (N x 2).
        filenames (list of str): List of filenames corresponding to the embeddings.
        csv_path (str): Path to the CSV file containing the labels.
        label_column (str): The column name in the CSV to use as labels.
        gridsize (int): The size of the hexagonal grid.
        save_path (str): Path to save the resulting plot.
    """
    # Load labels from the CSV
    filename_to_label = {}
    with open(csv_path, mode="r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            filename = row["Filename"]  # Filename in the CSV
            label = float(row[label_column])  # Corresponding label
            filename_to_label[filename] = label

    # Normalize filenames from the dataset by removing directory paths and extensions
    normalized_filenames = [os.path.splitext(os.path.basename(fname))[0] for fname in filenames]

    # Match labels to embeddings
    labels = []
    missing_files = []
    for fname in normalized_filenames:
        if fname in filename_to_label:
            labels.append(filename_to_label[fname])
        else:
            labels.append(None)
            missing_files.append(fname)

    # Warn about missing files
    if missing_files:
        print(f"Warning: {len(missing_files)} filenames do not have matching labels in the CSV.")
        print("Examples of missing files:", missing_files[:10])

    # Filter out embeddings and filenames without labels
    filtered_embeddings = [embedding for embedding, label in zip(embeddings_2d, labels) if label is not None]
    filtered_labels = [label for label in labels if label is not None]

    if not filtered_embeddings:
        raise ValueError("No valid embeddings found after filtering. Check filenames and CSV.")

    # Convert filtered embeddings to numpy array
    filtered_embeddings = np.array(filtered_embeddings)

    # Plot hexbin
    plt.figure(figsize=(10, 8))
    hexbin = plt.hexbin(
        filtered_embeddings[:, 0],
        filtered_embeddings[:, 1],
        C=filtered_labels,
        gridsize=gridsize,
        cmap="viridis",
        reduce_C_function=np.mean,
    )
    plt.colorbar(hexbin, label=label_column)
    plt.title(f"Hexbin Plot with Labels ({label_column})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(save_path, dpi=300)
    print(f"Hexbin plot saved to {save_path}")