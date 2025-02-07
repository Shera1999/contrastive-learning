import copy
import yaml
import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

# Load Configuration
model_config_path = "configs/model_config.yaml"
with open(model_config_path, "r") as file:
    model_config = yaml.safe_load(file)

dino_config = model_config["model"].get("dino", {})

# Load Training Parameters from main_config.yaml
main_config_path = "configs/main_config.yaml"
with open(main_config_path, "r") as file:
    main_config = yaml.safe_load(file)

class DINOModel(pl.LightningModule):
    def __init__(self, learning_rate=None, max_epochs=None):
        super().__init__()

        # Load training parameters
        self.learning_rate = learning_rate if learning_rate else dino_config.get("learning_rate", 0.001)
        self.max_epochs = max_epochs if max_epochs else main_config["training"]["max_epochs"]

        # Select Backbone
        backbone_name = dino_config.get("backbone", "resnet18")
        if backbone_name == "dino_vits16":
            self.student_backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
            input_dim = self.student_backbone.embed_dim
        else:
            resnet = getattr(torchvision.models, backbone_name)(pretrained=False)
            self.student_backbone = nn.Sequential(*list(resnet.children())[:-1])
            input_dim = 512  # Default for ResNet-based models

        # Define Projection Heads
        self.student_head = DINOProjectionHead(
            input_dim,
            dino_config.get("hidden_dim", 512),
            64,
            dino_config.get("output_dim", 2048),
            freeze_last_layer=dino_config.get("freeze_last_layer", 1),
        )

        self.teacher_backbone = copy.deepcopy(self.student_backbone)
        self.teacher_head = DINOProjectionHead(
            input_dim,
            dino_config.get("hidden_dim", 512),
            64,
            dino_config.get("output_dim", 2048),
        )

        # Disable gradient updates for the teacher model
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        # Loss function
        self.criterion = DINOLoss(
            output_dim=dino_config.get("output_dim", 2048),
            warmup_teacher_temp_epochs=dino_config.get("warmup_teacher_temp_epochs", 5),
        )

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        # Update momentum for teacher network
        momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        # Process batch
        views = batch[0]  # DINO uses multiple augmented views
        views = [view.to(self.device) for view in views]
        global_views = views[:2]

        # Compute teacher and student outputs
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]

        # Compute loss
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True)
        return loss

    def on_after_backward(self):
        """Prevent large gradient updates on the last layer during early training."""
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
