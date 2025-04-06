from lightly.transforms.byol_transform import BYOLTransform, BYOLView1Transform, BYOLView2Transform

def get_byol_transform(config, input_size):
    view1 = BYOLView1Transform(
        input_size=input_size,
        cj_prob=config["cj_prob"],
        cj_strength=config["cj_strength"],
        cj_bright=config["cj_bright"],
        cj_contrast=config["cj_contrast"],
        cj_sat=config["cj_sat"],
        cj_hue=config["cj_hue"],
        min_scale=config["min_scale"],
        random_gray_scale=config["random_gray_scale"],
        gaussian_blur=config["view1"]["gaussian_blur"],
        solarization_prob=config["view1"]["solarization_prob"],
        kernel_size=config.get("kernel_size", None),
        sigmas=tuple(config["sigmas"]),
        vf_prob=config["vf_prob"],
        hf_prob=config["hf_prob"],
        rr_prob=config["rr_prob"],
        rr_degrees=config["rr_degrees"]
    )

    view2 = BYOLView2Transform(
        input_size=input_size,
        cj_prob=config["cj_prob"],
        cj_strength=config["cj_strength"],
        cj_bright=config["cj_bright"],
        cj_contrast=config["cj_contrast"],
        cj_sat=config["cj_sat"],
        cj_hue=config["cj_hue"],
        min_scale=config["min_scale"],
        random_gray_scale=config["random_gray_scale"],
        gaussian_blur=config["view2"]["gaussian_blur"],
        solarization_prob=config["view2"]["solarization_prob"],
        kernel_size=config.get("kernel_size", None),
        sigmas=tuple(config["sigmas"]),
        vf_prob=config["vf_prob"],
        hf_prob=config["hf_prob"],
        rr_prob=config["rr_prob"],
        rr_degrees=config["rr_degrees"]
    )

    return BYOLTransform(view_1_transform=view1, view_2_transform=view2)
