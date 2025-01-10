import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models.simclr import SimCLRModel
from data.data_loader import dataloader_train
import torch

# Set Tensor Core Precision
torch.set_float32_matmul_precision("medium")

# Load Configuration Files
config_path = "configs/main_config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Training Parameters
max_epochs = config["training"]["max_epochs"]
device = "auto" if config["training"]["device"] == "gpu" else config["training"]["device"]
learning_rate = config["training"].get("learning_rate", 6e-2)  # Default to 6e-2 if not set

# Initialize Model
model = SimCLRModel(learning_rate=learning_rate, max_epochs=max_epochs)

# Checkpointing
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss_ssl",
    dirpath="checkpoints/",
    filename="simclr-{epoch:02d}-{train_loss_ssl:.2f}",
    save_top_k=3,
    save_last=True,
    mode="min",
)

# Trainer
trainer = pl.Trainer(
    max_epochs=max_epochs,
    accelerator=device,
    devices=1,
    callbacks=[checkpoint_callback],
    enable_progress_bar=True,
    enable_model_summary=False,
    log_every_n_steps=1,
    precision="bf16-mixed",
)

# Train Model
trainer.fit(model, dataloader_train)

# Save the final trained model
torch.save(model.state_dict(), "checkpoints/final_model.pth")
print("Final model saved as 'checkpoints/final_model.pth'")
