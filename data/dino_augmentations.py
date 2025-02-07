import yaml
from lightly.transforms.dino_transform import DINOTransform

def load_dino_config():
    """Loads DINO-specific augmentation configuration from YAML file."""

    # Load `augmentations_config.yaml`
    with open("configs/augmentations_config.yaml", "r") as file:
        augmentations_config = yaml.safe_load(file)

    # Load `main_config.yaml` to get training parameters
    with open("configs/main_config.yaml", "r") as file:
        main_config = yaml.safe_load(file)

    dino_config = augmentations_config["augmentations"].get("dino_params", {})

    # Ensure rr_degrees is always a tuple of length 2
    rr_degrees = dino_config.get("rr_degrees", (-10, 10))  # Default to (-10, 10)
    if isinstance(rr_degrees, (int, float)):  
        rr_degrees = (-abs(rr_degrees), abs(rr_degrees))  # Convert to tuple

    return {
        "global_crop_size": dino_config.get("global_crop_size", 224),
        "global_crop_scale": tuple(dino_config.get("global_crop_scale", [0.4, 1.0])),
        "local_crop_size": dino_config.get("local_crop_size", 96),
        "local_crop_scale": tuple(dino_config.get("local_crop_scale", [0.05, 0.4])),
        "n_local_views": dino_config.get("n_local_views", 6),
        "hf_prob": dino_config.get("hf_prob", 0.5),
        "vf_prob": dino_config.get("vf_prob", 0.0),
        "rr_prob": dino_config.get("rr_prob", 0.0),
        "rr_degrees": rr_degrees,  # Ensure correct tuple format
        "cj_prob": dino_config.get("cj_prob", 0.8),
        "cj_strength": dino_config.get("cj_strength", 0.5),
        "cj_bright": dino_config.get("cj_bright", 0.8),
        "cj_contrast": dino_config.get("cj_contrast", 0.8),
        "cj_sat": dino_config.get("cj_sat", 0.4),
        "cj_hue": dino_config.get("cj_hue", 0.2),
        "random_gray_scale": dino_config.get("random_gray_scale", 0.2),
        "gaussian_blur": tuple(dino_config.get("gaussian_blur", [1.0, 0.1, 0.5])),
        "kernel_size": dino_config.get("kernel_size", None),
        "sigmas": tuple(dino_config.get("sigmas", [0.1, 2])),
        "solarization_prob": dino_config.get("solarization_prob", 0.2),
    }

def get_dino_transform():
    """Returns a DINOTransform with customizable parameters."""
    dino_params = load_dino_config()
    return DINOTransform(**dino_params)
