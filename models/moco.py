import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import yaml
import copy

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum, batch_shuffle, batch_unshuffle

# Load config
with open("configs/model_config.yaml", "r") as file:
    config = yaml.safe_load(file)

moco_config = config["model"]["moco"]
max_epochs = config.get("training", {}).get("max_epochs", 100)

class MoCoModel(pl.LightningModule):
    def __init__(self, learning_rate=None, max_epochs=max_epochs):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate if learning_rate else moco_config.get("learning_rate", 0.06)
        self.max_epochs = max_epochs

        # Backbone
        backbone = getattr(torchvision.models, moco_config["backbone"])(pretrained=False)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection_head = MoCoProjectionHead(moco_config["input_dim"], moco_config["proj_hidden_dim"], moco_config["out_dim"])

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Loss
        self.criterion = NTXentLoss(
            temperature=moco_config["temperature"],
            memory_bank_size=(moco_config["memory_bank_size"], moco_config["out_dim"])
        )

    def forward(self, x):
        return self.projection_head(self.backbone(x).flatten(start_dim=1))

    def forward_momentum(self, x):
        with torch.no_grad():
            return self.projection_head_momentum(self.backbone_momentum(x).flatten(start_dim=1))

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        update_momentum(self.backbone, self.backbone_momentum, moco_config["momentum"])
        update_momentum(self.projection_head, self.projection_head_momentum, moco_config["momentum"])

        q = self.forward(x_q)
        k, shuffle = batch_shuffle(x_k)
        k = self.forward_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs)
        return [optimizer], [scheduler]
