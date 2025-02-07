import yaml
from lightly.transforms.simclr_transform import SimCLRTransform

def load_simclr_config():
    """Loads SimCLR-specific augmentation configuration from YAML files."""

    # Load augmentations_config.yaml
    with open("configs/augmentations_config.yaml", "r") as file:
        augmentations_config = yaml.safe_load(file)

    # Load main_config.yaml for training settings
    with open("configs/main_config.yaml", "r") as file:
        main_config = yaml.safe_load(file)

    simclr_config = augmentations_config["augmentations"].get("simclr_params", {})

    # Ensure rr_degrees is always a tuple of length 2
    rr_degrees = simclr_config.get("rr_degrees", (-10, 10))  # Default to (-10, 10)
    if isinstance(rr_degrees, (int, float)):  
        rr_degrees = (-abs(rr_degrees), abs(rr_degrees))  # Convert to tuple

    return {
        "input_size": main_config["training"]["input_size"],
        "cj_prob": simclr_config.get("cj_prob", 0.8),
        "cj_strength": simclr_config.get("cj_strength", 1.0),
        "cj_bright": simclr_config.get("cj_bright", 0.8),
        "cj_contrast": simclr_config.get("cj_contrast", 0.8),
        "cj_sat": simclr_config.get("cj_sat", 0.8),
        "cj_hue": simclr_config.get("cj_hue", 0.2),
        "min_scale": simclr_config.get("min_scale", 0.08),
        "random_gray_scale": simclr_config.get("random_gray_scale", 0.2),
        "gaussian_blur": simclr_config.get("gaussian_blur", 0.5),
        "kernel_size": simclr_config.get("kernel_size", None),
        "sigmas": tuple(simclr_config.get("sigmas", (0.1, 2))),
        "vf_prob": simclr_config.get("vf_prob", 0.0),
        "hf_prob": simclr_config.get("hf_prob", 0.5),
        "rr_prob": simclr_config.get("rr_prob", 0.5),
        "rr_degrees": rr_degrees,  # Ensuring tuple format
    }

def get_simclr_transform():
    """Returns a SimCLRTransform with customizable parameters."""
    simclr_params = load_simclr_config()
    return SimCLRTransform(**simclr_params)
