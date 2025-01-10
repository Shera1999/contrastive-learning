import yaml
import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as V2
import numpy as np
from PIL import Image
import random
from lightly.transforms.multi_view_transform import MultiViewTransform

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

        # Define individual augmentations
        self.augmentations_list = []

        if self.augmentations["shape_invariances"]["random_crop"]["enabled"]:
            self.augmentations_list.append(
                T.RandomResizedCrop(self.input_size, scale=(0.2, 1.0))
            )
        
        if self.augmentations["shape_invariances"]["random_horizontal_flip"]["enabled"]:
            prob = self.augmentations["shape_invariances"]["random_horizontal_flip"]["prob"]
            self.augmentations_list.append(T.RandomHorizontalFlip(prob))

        if self.augmentations["shape_invariances"]["random_vertical_flip"]["enabled"]:
            prob = self.augmentations["shape_invariances"]["random_vertical_flip"]["prob"]
            self.augmentations_list.append(T.RandomVerticalFlip(prob))

        if self.augmentations["shape_invariances"]["random_rotation"]["enabled"]:
            degrees = self.augmentations["shape_invariances"]["random_rotation"]["degrees"]
            self.augmentations_list.append(T.RandomRotation(degrees))

        if self.augmentations["geometric_transformations"]["zoom"]["enabled"]:
            scale_range = self.augmentations["geometric_transformations"]["zoom"]["scale"]
            self.augmentations_list.append(T.RandomAffine(degrees=0, scale=scale_range))

        if self.augmentations["geometric_transformations"]["affine_translation"]["enabled"]:
            translate = self.augmentations["geometric_transformations"]["affine_translation"]["translate_percent"]
            self.augmentations_list.append(T.RandomAffine(degrees=0, translate=translate))

        if self.augmentations["texture_invariances"]["gaussian_blur"]["enabled"]:
            sigma_range = self.augmentations["texture_invariances"]["gaussian_blur"]["sigma"]
            self.augmentations_list.append(T.GaussianBlur(kernel_size=(5, 5), sigma=sigma_range))

        if self.augmentations["texture_invariances"]["noise"]["enabled"]:
            self.augmentations_list.append(GaussianNoise())  # Custom noise class

        self.augmentations_list.extend([
            V2.ToImage(),
            V2.ToDtype(torch.float32, scale=True),
            V2.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

        # Compose transformations
        self.view_transform = T.Compose(self.augmentations_list)

        # MultiViewTransform: Generates two different augmented views for SimCLR
        self.multi_view_transform = MultiViewTransform(transforms=[self.view_transform, self.view_transform])

    def apply_augmentations(self, image):
        """Applies two different augmentations to a single image for SimCLR."""
        return self.multi_view_transform(image)