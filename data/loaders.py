import os
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2  # Import v2 for new transforms
from lightly.data import LightlyDataset
from data.preprocess import load_transforms  # Ensure this uses updated preprocess.py

def load_config(main_config_path, dataset_config_path):
    """Load global and dataset configurations."""
    with open(main_config_path, "r") as main_file:
        main_config = yaml.safe_load(main_file)
    with open(dataset_config_path, "r") as dataset_file:
        dataset_config = yaml.safe_load(dataset_file)
    
    # Merge configurations
    config = {**main_config, **dataset_config}
    print("Loaded Config:", config)
    return config


def get_dataset_loaders(model_name, model_config, dataset_config):
    """
    Prepares the DataLoader objects for training and testing.

    Args:
        model_name (str): Name of the model (e.g., "simclr").
        model_config (dict): Model-specific configuration dictionary.
        dataset_config (dict): Dataset-specific configuration dictionary.

    Returns:
        tuple: Train DataLoader, Test DataLoader
    """
    data_path = dataset_config["dataset"]["data_path"]
    test_split = dataset_config["dataset"]["test_split"]

    # Load the dataset
    dataset = LightlyDataset(input_dir=data_path, transform=None)

    # Split dataset indices into train and test
    dataset_indices = np.arange(len(dataset))
    train_indices, test_indices = train_test_split(
        dataset_indices, test_size=test_split, random_state=42
    )

    # Load transforms based on model
    train_transform = load_transforms(model_config, model_name, "train_transforms")
    test_transform = load_transforms(model_config, model_name, "test_transforms")

    # Create subsets with transforms
    train_dataset = Subset(dataset, train_indices)
    train_dataset.dataset.transform = train_transform

    test_dataset = Subset(dataset, test_indices)
    test_dataset.dataset.transform = test_transform

    # Create DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=dataset_config["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=dataset_config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=dataset_config["num_workers"],
    )

    return train_loader, test_loader