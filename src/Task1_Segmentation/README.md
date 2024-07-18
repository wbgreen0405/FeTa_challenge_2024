# Task 1 - Segmentation

This directory contains the code for the segmentation task of the Fetal Tissue Annotation Challenge. The goal of this task is to segment different tissues in T2-weighted fetal brain images.

## Directory Structure

- `__main__.py`: Main script to run inference on a single subject.
- `models/segmentation_model.py`: Contains the model architecture for segmentation.
- `scripts/preprocess.py`: Functions for loading and preprocessing images.
- `scripts/train.py`: Script to train the segmentation model.
- `scripts/evaluate.py`: Script to evaluate the model and visualize predictions.
- `scripts/inference.py`: Inference script to predict segmentations from a single subject.
- `scripts/utils.py`: Utility functions for creating datasets and saving images.
- `inference.sh`: Shell script to run the inference script with specified arguments.
