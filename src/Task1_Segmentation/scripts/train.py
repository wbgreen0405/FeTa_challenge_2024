import os
import glob
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import backend as K
import albumentations as A
from scripts.preprocess import load_nifti_image, load_nifti_label
from scripts.utils import create_tf_dataset
from models.segmentation_model import unet_3plus_model

# Define additional functions and classes here...

def k_fold_cross_validation(train_paths, img_size, num_classes, batch_size, k=5):
    # ... (same code as before)
    return val_scores, histories, fold_indices, models

def run_multiple_kfolds(train_paths, val_paths, img_size, num_classes, batch_size, k=5, num_runs=1):
    # ... (same code as before)
    return all_val_scores, all_histories, all_fold_indices, all_models

# Main training script here...
