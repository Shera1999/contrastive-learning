import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import yaml

class BaseContrastiveModel(nn.Module):
    def __init__(self, model_config_path, main_config_path):
        """Base model for contrastive learning."""
        super().__init__()

        # Load config files
        with open(model_config_path, "r") as file:
            self.model_config = yaml.safe_load(file)
        
        with open(main_config_path, "r") as file:
            self.main_config = yaml.safe_load(file)

        # Get selected model
        self.model_name = self.model_config["model"]["selected_model"]
        self.model_params = self.model_config["model"][self.model_name]

        # Extract training parameters
        self.batch_size = self.main_config["training"]["batch_size"]
        self.input_size = self.main_config["training"]["input_size"]
        self.num_workers = self.main_config["training"]["num_workers"]
        self.device = self.main_config["training"]["device"]

        # Normalization parameters
        self.norm_mean = self.main_config["normalization"]["mean"]
        self.norm_std = self.main_config["normalization"]["std"]

        # Get backbone
        self.backbone = self.get_backbone(self.model_params["backbone"])

    def get_backbone(self, backbone_name):
        """Load backbone without the classification head."""
        backbone_dict = {
            "resnet18": models.resnet18(pretrained=False),
            "resnet34": models.resnet34(pretrained=False),
            "resnet50": models.resnet50(pretrained=False),
        }
        if backbone_name not in backbone_dict:
            raise ValueError(f"Invalid backbone: {backbone_name}")
        
        backbone = backbone_dict[backbone_name]
        return nn.Sequential(*list(backbone.children())[:-1])  # Remove classification head

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            lr=self.main_config["training"]["learning_rate"], 
            momentum=0.9, 
            weight_decay=5e-4
        )
        
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.main_config["training"]["max_epochs"]),
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler} 