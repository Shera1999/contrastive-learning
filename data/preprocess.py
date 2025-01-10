# preprocess.py

from torchvision.transforms import v2
import yaml

def load_transforms(model_config, model_name, transform_type="train_transforms"):
    """
    Loads transforms dynamically from the model configuration file.

    Args:
        model_config (dict): Model-specific configuration.
        model_name (str): Name of the model (e.g., "simclr").
        transform_type (str): Type of transform ("train_transforms" or "test_transforms").

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    import torch  # Ensure torch is available in the lambda context

    transform_list = []
    for transform in model_config[model_name][transform_type]:
        for name, params in transform.items():
            if name == "Lambda":
                function = eval(params["function"], {"torch": torch})
                transform_list.append(v2.Lambda(function))
            elif name == "RandomApply":
                sub_transforms = []
                for sub_transform in params["transforms"]:
                    for sub_name, sub_params in sub_transform.items():
                        sub_class = getattr(v2, sub_name, None) or getattr(transforms, sub_name, None)
                        sub_transforms.append(sub_class(**sub_params) if sub_params else sub_class())
                transform_list.append(v2.RandomApply(sub_transforms, p=params["probability"]))
            else:
                transform_class = getattr(v2, name, None) or getattr(transforms, name, None)
                if not transform_class:
                    raise AttributeError(f"Invalid transform: {name}")
                transform_list.append(transform_class(**params) if params else transform_class())

    return v2.Compose(transform_list)
