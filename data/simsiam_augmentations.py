import yaml
from lightly.transforms.simsiam_transform import SimSiamTransform

def get_simsiam_transform(config_path="configs/augmentations_config.yaml", main_config_path="configs/main_config.yaml"):
    """Build SimSiam augmentations from config YAML."""
    with open(config_path, "r") as file:
        aug_config = yaml.safe_load(file)

    with open(main_config_path, "r") as file:
        main_config = yaml.safe_load(file)

    simsiam_params = aug_config["augmentations"]["simsiam_params"]
    normalization = aug_config["normalization"]

    return SimSiamTransform(
        input_size=simsiam_params["input_size"],
        cj_prob=simsiam_params["cj_prob"],
        cj_strength=simsiam_params["cj_strength"],
        cj_bright=simsiam_params["cj_bright"],
        cj_contrast=simsiam_params["cj_contrast"],
        cj_sat=simsiam_params["cj_sat"],
        cj_hue=simsiam_params["cj_hue"],
        min_scale=simsiam_params["min_scale"],
        random_gray_scale=simsiam_params["random_gray_scale"],
        gaussian_blur=simsiam_params["gaussian_blur"],
        kernel_size=simsiam_params["kernel_size"],
        sigmas=tuple(simsiam_params["sigmas"]),
        vf_prob=simsiam_params["vf_prob"],
        hf_prob=simsiam_params["hf_prob"],
        rr_prob=simsiam_params["rr_prob"],
        rr_degrees=simsiam_params["rr_degrees"],
        normalize={"mean": normalization["mean"], "std": normalization["std"]},
    )
