# Task 2 - Biometry

This directory contains the code for the biometry task of the Fetal Tissue Annotation Challenge. The goal of this task is to predict biometric measurements from T2-weighted fetal brain images.

## Directory Structure

- `__main__.py`: Main script to run inference on a single subject.
- `models/biometry_model.py`: Contains the model architecture for biometry measurement prediction.
- `scripts/preprocess.py`: Functions for loading and preprocessing images.
- `scripts/train.py`: Script to train the biometry measurement prediction model.
- `scripts/evaluate.py`: Script to evaluate the model and visualize predictions.
- `scripts/utils.py`: Utility functions for loading data and other tasks.
- `test.py`: Inference script to predict biometry measurements from a single subject.
- `inference.sh`: Shell script to run the inference script with specified arguments.
