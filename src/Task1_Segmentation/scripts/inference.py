import os
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from scripts.preprocess import load_nifti_image
from scripts.utils import save_nifti_image

def tta_predict(models, image, num_augmentations=5):
    # ... (same code as before)
    return np.argmax(final_avg_pred, axis=-1).astype(np.uint8)

def run_inference_on_test_images(models, test_image_paths, output_dir):
    img_size = (256, 256)
    
    for t2w_image in test_image_paths:
        print(f"Processing image: {t2w_image}")
        img = load_nifti_image(t2w_image, img_size)
        print(f"Loaded image shape: {img.shape}, Image dtype: {img.dtype}")
        
        # Predict segmentation using averaged predictions from all models
        predicted_seg = tta_predict(models, img)
        print(f"Predicted segmentation shape: {predicted_seg.shape}, Predicted dtype: {predicted_seg.dtype}")
        
        # Resize predicted segmentation to match the original input size
        predicted_seg_resized = cv2.resize(predicted_seg, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        print(f"Resized predicted segmentation shape: {predicted_seg_resized.shape}")
        
        # Save the segmentation
        output_seg = os.path.join(output_dir, os.path.basename(t2w_image).replace('.nii', '_segmentation.nii.gz'))
        save_nifti_image(predicted_seg_resized, t2w_image, output_seg)
        print(f"Segmentation saved to {output_seg}")
        
        # Visualize the output segmentation
        visualize_nifti_with_labels(output_seg)

def load_all_models(model_dir):
    model_files = glob.glob(os.path.join(model_dir, '*.keras'))
    models = []
    for model_file in model_files:
        models.append(tf.keras.models.load_model(model_file, custom_objects={ 'boundary_aware_loss': boundary_aware_loss,'dice_coef': dice_coef, 'iou_coef': iou_coef}))
    return models

# Define visualization function with labels
def visualize_nifti_with_labels(file_path, num_slices=5):
    # ... (same code as before)
