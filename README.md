# **Contrastive Learning**

This repository contains a self-supervised learning pipeline for Imagae classification using SimCLR and DINO models. The project includes data preprocessing, training, embedding generation, and visualization of embeddings using UMAP and nearest neighbors.

---

## **Project Structure**
```
 CL/
│── configs/                  # Configuration files (YAML format)
│    ├── dataset_config.yaml        # Dataset paths and parameters
│    ├── model_config.yaml          # Model selection and hyperparameters
│    ├── main_config.yaml           # Training configurations
│    ├── augmentations_config.yaml  # Augmentations configurations
│
│── data/                     # Data processing scripts
│    ├── data_loader.py          # Loads training and test datasets
│    ├── data_augmentor.py       # Applies data augmentations
│    ├── simclr_augmentations.py # Augmentations for SimCLR
│    ├── dino_augmentations.py   # Augmentations for DINO
│
│── models/                   # Model implementations
│    ├── simclr.py               # SimCLR model definition
│    ├── dino.py                 # DINO model definition
│
│── postprocessing/            # Embedding analysis and visualization
│    ├── generate_embeddings.py  # Extracts embeddings from trained models
│    ├── plot_umap.py            # UMAP visualization of embeddings
│    ├── plot_knn.py             # Nearest neighbor visualization
│    ├── plot_hexbin.py          # 2D hexbin histogram visualization
│
│──  checkpoints/                 # Model checkpoints
│──  logs/                        # Training logs
│──  datasets/                    # Dataset directory
│── main.sh                       # Shell script for training and embedding generation
│── main.ipynb                    # Jupyter Notebook for postprocessing and visualization
│── README.md                     # Project documentation
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
- **`configs/model_config.yaml`** → Choose between `simclr` and `dino`
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
  selected_model: "selected_model"
```
Here you can write the name of the model you want to be using, eg; simclr, dino


```yaml
# configs/augmentations_config.yaml
augmentations:
  use_custom: True
  use_simclr: False
  use_dino: False
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

---

## **Future Improvements**
-  Support additional self-supervised models (e.g., MoCo, BYOL)
-  Improve embedding quality with different backbone architectures
-  Extend label-based visualization techniques

