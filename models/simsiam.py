import torch
import torch.nn as nn
import torchvision
import yaml
import pytorch_lightning as pl

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

# Load SimSiam config
with open("configs/model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

simsiam_config = config["model"]["simsiam"]


class SimSiamModel(pl.LightningModule):
    def __init__(self, learning_rate=None, max_epochs=None):
        super().__init__()
        self.save_hyperparameters()

        # Configs
        self.learning_rate = learning_rate if learning_rate else simsiam_config.get("learning_rate", 0.05)
        self.max_epochs = max_epochs

        # Backbone
        backbone = getattr(torchvision.models, simsiam_config["backbone"])(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Projection + Prediction heads
        self.projection_head = SimSiamProjectionHead(
            simsiam_config["input_dim"],
            simsiam_config["proj_hidden_dim"],
            simsiam_config["out_dim"]
        )
        self.prediction_head = SimSiamPredictionHead(
            simsiam_config["out_dim"],
            simsiam_config["pred_hidden_dim"],
            simsiam_config["out_dim"]
        )

        # Loss
        self.criterion = NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        return z.detach(), p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0 = x0.to(self.device)
        x1 = x1.to(self.device)
        z0, p0 = self(x0)
        z1, p1 = self(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        return optimizer
