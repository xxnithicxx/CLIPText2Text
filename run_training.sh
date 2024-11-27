#!/bin/bash

# Activate virtual environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate MLC # replace myenv with your environment name

# Run training script
python scripts/train.py
