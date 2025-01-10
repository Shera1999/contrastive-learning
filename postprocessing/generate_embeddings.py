import torch
import numpy as np
import umap
from sklearn.preprocessing import normalize
from models.simclr import SimCLRModel
from data.data_loader import dataloader_test
import os

def load_model(model_path="checkpoints/final_model.pth", device="cuda" if torch.cuda.is_available() else "cpu"):
    """Loads the trained model from file."""
    model = SimCLRModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f" Loaded model from {model_path}")
    return model

def generate_embeddings(model, dataloader, save_path="embeddings.npy"):
    """Generates embeddings and saves them to a file."""
    embeddings = []
    filenames = []
    device = model.device

    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb.cpu())
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    np.save(save_path, embeddings)
    print(f" Embeddings saved to {save_path}")
    return embeddings, filenames

def generate_umap_projection(embeddings, save_path="embeddings_2d.npy"):
    """Applies UMAP and saves the 2D embeddings."""
    reducer = umap.UMAP(n_components=2, random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)
    np.save(save_path, embeddings_2d)
    print(f" UMAP 2D Embeddings saved to {save_path}")
    return embeddings_2d

if __name__ == "__main__":
    model = load_model()
    embeddings, filenames = generate_embeddings(model, dataloader_test)
    embeddings_2d = generate_umap_projection(embeddings)
