import yaml
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.transforms.utils import IMAGENET_NORMALIZE


def get_moco_transform():
    with open("configs/augmentations_config.yaml", "r") as file:
        config = yaml.safe_load(file)

    moco_params = config["augmentations"]["moco_params"]

    return MoCoV2Transform(
        input_size=moco_params.get("input_size", 224),
        cj_prob=moco_params.get("cj_prob", 0.8),
        cj_strength=moco_params.get("cj_strength", 1.0),
        cj_bright=moco_params.get("cj_bright", 0.4),
        cj_contrast=moco_params.get("cj_contrast", 0.4),
        cj_sat=moco_params.get("cj_sat", 0.4),
        cj_hue=moco_params.get("cj_hue", 0.1),
        min_scale=moco_params.get("min_scale", 0.2),
        random_gray_scale=moco_params.get("random_gray_scale", 0.2),
        gaussian_blur=moco_params.get("gaussian_blur", 0.5),
        kernel_size=moco_params.get("kernel_size", None),
        sigmas=tuple(moco_params.get("sigmas", [0.1, 2])),
        vf_prob=moco_params.get("vf_prob", 0.0),
        hf_prob=moco_params.get("hf_prob", 0.5),
        rr_prob=moco_params.get("rr_prob", 0.0),
        rr_degrees=moco_params.get("rr_degrees", None),
        normalize=config.get("normalization", IMAGENET_NORMALIZE),
    )
