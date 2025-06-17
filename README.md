# **Contrastive Learning**

This repository contains a self-supervised learning pipeline for Imagae classification using SimCLR and DINO models. The project includes data preprocessing, training, embedding generation, and visualization of embeddings using UMAP and nearest neighbors.

---

## **Project Structure**
```
CL/
│── configs/
│    ├── dataset_config.yaml         # Dataset paths and parameters
│    ├── model_config.yaml           # Model selection and hyperparameters
│    ├── main_config.yaml            # Training configurations
│    ├── augmentations_config.yaml  # Augmentations configurations
│
│── data/
│    ├── data_loader.py              # Loads training and test datasets
│    ├── data_augmentor.py           # Applies data augmentations
│    ├── simclr_augmentations.py     # SimCLR transforms
│    ├── dino_augmentations.py       # DINO transforms
│    ├── simsiam_augmentations.py    # SimSiam transforms
│    ├── moco_augmentations.py       # MoCo transforms
│    ├── byol_augmentations.py       # BYOL transforms
│
│── models/
│    ├── simclr.py                   # SimCLR model
│    ├── dino.py                     # DINO model
│    ├── simsiam.py                  # SimSiam model
│    ├── moco.py                     # MoCo model
│    ├── byol.py                     # BYOL model
│
│── postprocessing/
│    ├── generate_embeddings.py      # Extract embeddings
│    ├── plot_umap.py                # UMAP projection
│    ├── plot_knn.py                 # KNN visualization
│    ├── plot_grid.py                # Grid-based UMAP image plot
│    ├── plot_hexbin.py              # 2D hexbin histogram
│
│── checkpoints/                     # Saved models
│── logs/                            # Training logs
│── datasets/                        # Dataset folder
│── main.sh                          # End-to-end training + embedding
│── main.ipynb                       # Postprocessing and visualization
│── README.md                        # Project documentation

```

---

## **Setup Instructions**
### **1. Clone the Repository**
```bash
git clone https://github.com/Shera1999/contrastive-learning.git
cd CL/
```

### **2. Create and Activate a Virtual Environment**
```bash
python -m venv new_venv
source new_venv/bin/activate  # Linux/Mac
new_venv\Scripts\activate     # Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```
> If `requirements.txt` is not available, manually install key dependencies:
```bash
pip install numpy torch torchvision pytorch-lightning matplotlib umap-learn pandas scikit-learn Pillow yaml
```

### **4. Configure Dataset and Model Settings**
Modify the following configuration files as needed:

- **`configs/dataset_config.yaml`** → Set dataset paths
- **`configs/model_config.yaml`** → Choose from: `simclr`, `dino`, `simsiam`, `moco`, `byol`
- **`configs/main_config.yaml`** → Training parameters
- **`configs/main_config.yaml`** → Augmentations parameters where you can choose a custom augmentations with few parameters, or you can choose to use a set of augmentations defined by lightly for each model, and pick their values. 

---

## **Usage**
### **1. Train the Model**
Run the training and embedding generation script:
```bash
./main.sh
```
This will:
- Train the selected model (`simclr` or `dino`)
- Save model checkpoints in `checkpoints/`
- Generate and save embeddings in `embeddings.npy` and `embeddings_2d.npy`

### **2. Generate Embeddings Only**
If the model is already trained, you can generate embeddings separately:
```bash
python postprocessing/generate_embeddings.py
```

### **3. Postprocessing & Visualization**
Run the Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```
Inside the notebook, you can:
- **Plot the k nearest neighbors** of a selected number of random images
- **Visualize UMAP projection** of embeddings
- **Visualize the images on the UMAP** projections
- **Find nearest neighbors** for selected images
- **Plot hexbin histograms** for labels in UMAP space (in case you have a labeled dataset)

---

## **Features**
### **Model Training**
- Supports **SimCLR** and **DINO**
- Uses PyTorch Lightning for efficient training
- Configurable via `model_config.yaml`

### **Embeddings Extraction**
- Saves extracted embeddings from trained models
- Supports different backbone architectures

### **Visualization**
- **UMAP Projection**: 2D visualization of embeddings
- **K-Nearest Neighbors (KNN)**: Find similar images based on embeddings
- **Hexbin Histogram**: Display label distribution in 2D UMAP space (in case of having a labeled dataset)

---

## **Example Commands**
### **Configuration Files**

```yaml
# configs/model_config.yaml
model:
  selected_model: "simclr"
```


```yaml
# configs/augmentations_config.yaml
augmentations:
  use_custom: True
  use_simclr: False
  use_dino: False
  use_simsiam: False
  use_moco: False
  use_byol: False
```
or you can choose to use set of augmentations specifically desinged by lightly for each model, by setting use_model = True. After this, you can change each of their parameters model_params. 


```bash
./main.sh
```

### **Find Nearest Neighbors**
In **`main.ipynb`**, run:
```python
query_filename = "example.png"
plot_knn_for_specific_image(embeddings, filenames, query_filename, n_neighbors=5)
```

### **Plot Hexbin with Labels**
```python
csv_path = "example.csv"
label_column = "column1"

plot_hexbin_with_labels(
    embeddings_2d,
    filenames,
    csv_path=csv_path,
    label_column=label_column,
    gridsize=30,
    save_path="hexbin_with_labels.png",
)
```

---

## **Examples **

This section showcases examples of how the learned representations from contrastive models capture meaningful structure in image data. Below are UMAP projections and nearest neighbor visualizations for two different datasets: **jellyfish galaxies** and **X-ray maps of galaxy clusters**.

---

### **🪼 Jellyfish Galaxies**

**Data Source**:
[Zooniverse - Cosmological Jellyfish](https://www.zooniverse.org/projects/apillepich/cosmological-jellyfish)

These examples use galaxy cutouts from the Cosmological Jellyfish project to train a SimCLR model.

* **UMAP Projection**
  `UMAP_jellyfish.png` shows the 2D UMAP projection of the learned embedding space using SimCLR. The structure shows how galaxies with similar morphologies are grouped together in the learned representation.

* **Nearest Neighbors Visualization**
  `NN_jellyfish.jpg` presents the top-5 nearest neighbors for several query images. The overlaid scores indicate the model-inferred probability of each galaxy being a jellyfish. As shown, the retrieved neighbors not only look visually similar but also have high probability scores, confirming that the model effectively clusters jellyfish galaxies in the learned space.

---

### **🌌 Galaxy Clusters — TNG-Cluster X-ray Maps**

**Data Source**:
X-ray maps of galaxy clusters from the [TNG-Cluster simulations](https://www.tng-project.org/cluster/), using three projections across 8 snapshots.

* **UMAP Projection**
  `UMAP_X-ray.png` shows the learned embedding space using DINO on raw X-ray cluster maps. Different clusters and morphologies emerge in distinct regions of the UMAP plot.

* **Nearest Neighbors Visualization**
  `NN_X-ray.jpg` displays visually similar clusters retrieved via nearest neighbors in embedding space. The similarity of X-ray morphology among neighbors supports the model’s ability to capture meaningful visual representations.

* **Hexbin: Observables and Unobservables**
  `umap_observables.png` and `umap_unobservables.png` show 2D hexbin histograms of observable and unobservable physical properties (e.g., X-ray luminosity, merger stage) mapped onto the learned 2D UMAP space. The visible clustering patterns suggest that the model has captured underlying astrophysical structure in the data, even though it was trained without labels.

These insights form the basis for the next project:
👉 [CINN\_spline: Conditional Invertible Neural Networks for Physical Inference](https://github.com/Shera1999/CINN_spline)

---

