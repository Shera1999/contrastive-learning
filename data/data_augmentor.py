import yaml
import torch
from data.augmentations import Augmentations
from lightly.data import LightlyDataset
import torchvision.transforms.v2 as V2

class DataAugmentor:
    def __init__(self, augmentations_config, dataset_config, main_config):
        """Initializes the data augmentor using configurations."""
        # Load configuration files
        self.augmentations_config = self.load_yaml(augmentations_config)
        self.dataset_config = self.load_yaml(dataset_config)
        self.main_config = self.load_yaml(main_config)

        # Extract dataset and training parameters
        self.data_path = self.dataset_config["dataset"]["data_path"]
        self.batch_size = self.main_config["training"]["batch_size"]
        self.input_size = self.main_config["training"]["input_size"]
        self.num_workers = self.main_config["training"]["num_workers"]

        # Normalization parameters from augmentations_config
        self.norm_mean = self.augmentations_config["normalization"]["mean"]
        self.norm_std = self.augmentations_config["normalization"]["std"]

        # Initialize augmentations
        self.augmentor = Augmentations(augmentations_config, main_config)

    def load_yaml(self, config_path):
        """Loads a YAML configuration file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def get_train_transform(self):
        """Ensures SimCLR receives (x0, x1) tuple."""
        return self.augmentor.apply_augmentations

    def get_test_transform(self):
        """Returns the test transformation (resizing & normalization)."""
        return V2.Compose([
            V2.Resize((self.input_size, self.input_size)),
            V2.ToImage(),
            V2.ToDtype(torch.float32, scale=True),
            V2.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

    def get_datasets(self):
        """Creates train and test datasets with appropriate transformations."""
        train_transform = self.get_train_transform()
        test_transform = self.get_test_transform()

        dataset_train = LightlyDataset(input_dir=self.data_path, transform=train_transform)
        dataset_test = LightlyDataset(input_dir=self.data_path, transform=test_transform)

        return dataset_train, dataset_test

    def get_dataloaders(self):
        """Creates and returns DataLoaders for training and testing."""
        dataset_train, dataset_test = self.get_datasets()

        dataloader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers
        )

        return dataloader_train, dataloader_test
