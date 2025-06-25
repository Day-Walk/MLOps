#!/bin/bash
# This script runs the congestion prediction service.

# Navigate to the project root directory from the script's location
# This ensures that all paths loaded from .env or elsewhere are resolved correctly.
cd "$(dirname "$0")/../../../"

# Activate conda environment and run the Python script
echo "Activating conda environment..."
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate final-project

echo "Starting Python prediction script..."
python MLOps/app/services/predict_cong_service.py 