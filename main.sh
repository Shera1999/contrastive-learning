#!/bin/bash

# Activate the virtual environment
source /u/jshera/CL/new_venv/bin/activate

# Suppress all warnings but keep errors and info messages
export PYTHONWARNINGS="ignore"

# Run training and save logs
echo "Start Training..."

python /u/jshera/CL/models/train.py 2>&1 | tee training.log

echo " Training Completed."
echo " Training Completed. Logs saved in training.log."
echo " Checkpoints saved in 'checkpoints/'"
echo " Final model saved as 'checkpoints/final_model.pth'"

python postprocessing/generate_embeddings.py
echo " Embeddings saved as 'embeddings.npy' and 'embeddings_2d.npy'"