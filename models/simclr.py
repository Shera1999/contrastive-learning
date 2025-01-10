import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

class SimCLRModel(pl.LightningModule):
    def __init__(self, learning_rate=6e-2, max_epochs=100):
        super().__init__()

        # ResNet Backbone (Removing Classification Head)
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        h = self.backbone(x)  # Extract features
        h = h.flatten(start_dim=1)  # Flatten features
        z = self.projection_head(h)  # Projection Head
        return z

    def training_step(self, batch, batch_idx):
        """Ensures batch input is correctly unpacked"""
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)

        # Explicitly specify batch size
        self.log("train_loss_ssl", loss, prog_bar=True, on_epoch=True, batch_size=x0.size(0))
        return loss

    def configure_optimizers(self):
        """Uses SGD with CosineAnnealingLR."""
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return [optimizer], [scheduler]
