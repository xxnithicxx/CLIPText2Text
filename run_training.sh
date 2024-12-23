#!/bin/bash

# Activate virtual environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate CLIPText2Text # replace myenv with your environment name

# Run training script
python scripts/train.py
