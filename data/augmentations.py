import yaml
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as V2
import numpy as np
from PIL import Image
import random
from lightly.transforms.multi_view_transform import MultiViewTransform
from data.simclr_augmentations import get_simclr_transform
from data.dino_augmentations import get_dino_transform
from data.simsiam_augmentations import get_simsiam_transform
from data.moco_augmentations import get_moco_transform
from data.byol_augmentations import get_byol_transform

class GaussianNoise:
    """Applies random Gaussian noise to a tensor.

    The intensity of the noise is dependent on the mean of the pixel values.
    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)  # Random SNR between 4 and 8
        sigma = mu / snr
        noise = torch.normal(torch.zeros_like(sample), sigma)
        return sample + noise  # Adds noise while keeping image tensor format

class Augmentations:
    def __init__(self, config_path, main_config_path):
        """Load augmentation configurations from YAML files."""
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        with open(main_config_path, "r") as file:
            self.main_config = yaml.safe_load(file)

        self.augmentations = self.config["augmentations"]
        self.input_size = self.main_config["training"]["input_size"]

        # Normalization parameters
        self.norm_mean = self.config["normalization"]["mean"]
        self.norm_std = self.config["normalization"]["std"]

        # Select augmentation mode
        self.selected_transform = self.create_augmentations()

    def create_augmentations(self):
        """Creates and returns the selected augmentation strategy."""

        if self.augmentations.get("use_simclr", False):
            return get_simclr_transform()
        
        if self.augmentations.get("use_dino", False):
            return get_dino_transform()

        if self.augmentations.get("use_simsiam", False):
            return get_simsiam_transform()
        
        if self.augmentations.get("use_moco", False):
            return get_moco_transform()
        
        if self.augmentations.get("use_byol", False):
            return get_byol_transform(self.config["byol_params"], self.input_size)

        if self.augmentations.get("use_custom", False):
            return self.create_custom_augmentations()

        raise ValueError("Invalid augmentation settings. Enable one of 'use_simclr', 'use_dino', or 'use_custom' in augmentations_config.yaml.")
        
    # Define individual augmentations
    def create_custom_augmentations(self):
        augmentations_list = []

        if self.augmentations["shape_invariances"]["random_crop"]["enabled"]:
            augmentations_list.append(
                T.RandomResizedCrop(self.input_size, scale=(0.2, 1.0))
            )
        
        if self.augmentations["shape_invariances"]["random_horizontal_flip"]["enabled"]:
            prob = self.augmentations["shape_invariances"]["random_horizontal_flip"]["prob"]
            augmentations_list.append(T.RandomHorizontalFlip(prob))

        if self.augmentations["shape_invariances"]["random_vertical_flip"]["enabled"]:
            prob = self.augmentations["shape_invariances"]["random_vertical_flip"]["prob"]
            augmentations_list.append(T.RandomVerticalFlip(prob))

        if self.augmentations["shape_invariances"]["random_rotation"]["enabled"]:
            degrees = self.augmentations["shape_invariances"]["random_rotation"]["degrees"]
            augmentations_list.append(T.RandomRotation(degrees))

        if self.augmentations["geometric_transformations"]["zoom"]["enabled"]:
            scale_range = self.augmentations["geometric_transformations"]["zoom"]["scale"]
            augmentations_list.append(T.RandomAffine(degrees=0, scale=scale_range))

        if self.augmentations["geometric_transformations"]["affine_translation"]["enabled"]:
            translate = self.augmentations["geometric_transformations"]["affine_translation"]["translate_percent"]
            augmentations_list.append(T.RandomAffine(degrees=0, translate=translate))

        if self.augmentations["texture_invariances"]["gaussian_blur"]["enabled"]:
            sigma_range = self.augmentations["texture_invariances"]["gaussian_blur"]["sigma"]
            augmentations_list.append(T.GaussianBlur(kernel_size=(5, 5), sigma=sigma_range))

        if self.augmentations["texture_invariances"]["noise"]["enabled"]:
            augmentations_list.append(GaussianNoise())  # Custom noise class

        augmentations_list.extend([
            V2.ToImage(),
            V2.ToDtype(torch.float32, scale=True),
            V2.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

        return MultiViewTransform(transforms=[T.Compose(augmentations_list), T.Compose(augmentations_list)])

    def apply_augmentations(self, image):
        """Applies two different augmentations to a single image for SimCLR."""
        return self.selected_transform(image)