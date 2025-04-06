import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from models.simclr import SimCLRModel
from models.dino import DINOModel
from models.simsiam import SimSiamModel
from models.moco import MoCoModel
from models.byol import BYOLModel
from data.data_loader import dataloader_train
import torch

# Set Tensor Core Precision
torch.set_float32_matmul_precision("medium")

# Load Configuration Files
main_config_path = "configs/main_config.yaml"
model_config_path = "configs/model_config.yaml"

with open(main_config_path, "r") as file:
    main_config = yaml.safe_load(file)
with open(model_config_path, "r") as file: 
    model_config = yaml.safe_load(file)

# Ensure 'model' key exists
if "model" not in model_config:
    raise KeyError("Missing 'model' key in model_config.yaml")

# Get Selected Model
selected_model = model_config["model"].get("selected_model", "simclr")  # Default: simclr

# Training Parameters
max_epochs = main_config["training"]["max_epochs"]
device = "auto" if main_config["training"]["device"] == "gpu" else main_config["training"]["device"]

learning_rate = model_config["model"].get(selected_model, {}).get("learning_rate", 0.001)# Default to 0.001 if not set

# Initialize Model

# Dynamically Select Model and Pass Learning Rate
if selected_model == "simclr":
    model = SimCLRModel(learning_rate=learning_rate, max_epochs=max_epochs)
elif selected_model == "dino":
    model = DINOModel(learning_rate=learning_rate, max_epochs=max_epochs)
elif selected_model == "simsiam":
    model = SimSiamModel(learning_rate=learning_rate, max_epochs=max_epochs)
elif selected_model == "moco":
    model = MoCoModel(learning_rate=learning_rate, max_epochs=max_epochs)
elif selected_model == "byol":
    model = BYOLModel(learning_rate=learning_rate, max_epochs=max_epochs)
else:
    raise ValueError(f"Unknown model selected: {selected_model}")

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
