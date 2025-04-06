import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import yaml
import copy

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

# Load config
with open("configs/model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

byol_config = config["model"]["byol"]
max_epochs = config.get("training", {}).get("max_epochs", 100)


class BYOLModel(pl.LightningModule):
    def __init__(self, learning_rate=None, max_epochs=max_epochs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate if learning_rate else byol_config.get("learning_rate", 0.06)
        self.max_epochs = max_epochs

        backbone = getattr(torchvision.models, byol_config["backbone"])(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = BYOLProjectionHead(byol_config["input_dim"], byol_config["proj_hidden_dim"], byol_config["out_dim"])
        self.prediction_head = BYOLPredictionHead(byol_config["out_dim"], byol_config["pred_hidden_dim"], byol_config["out_dim"])

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y).detach()
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)

        # Momentum update
        update_momentum(self.backbone, self.backbone_momentum, byol_config["momentum"])
        update_momentum(self.projection_head, self.projection_head_momentum, byol_config["momentum"])

        p0 = self(x0)
        p1 = self(x1)
        z0 = self.forward_momentum(x0)
        z1 = self.forward_momentum(x1)

        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs)
        return [optimizer], [scheduler]
