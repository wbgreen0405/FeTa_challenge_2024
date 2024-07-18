import os
import argparse
import numpy as np
import nibabel as nib
from skimage.transform import resize
from tensorflow.keras.models import load_model
import pandas as pd

def load_and_preprocess_image(file_path, target_shape=(128, 128, 128)):
    img = nib.load(file_path)
    img_data = img.get_fdata()

    img_resized = resize(img_data, target_shape, mode='constant', anti_aliasing=True)
    img_norm = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))

    img_norm = np.repeat(img_norm[..., np.newaxis], 3, axis=-1)
    return img_norm

def predict_biometry(model, image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    pred = model.predict(image)
    return pred

def main(args):
    model = load_model('biometry_measurement_model.h5')
    
    t2w_image = load_and_preprocess_image(args.input)
    
    pred_biom = predict_biometry(model, t2w_image)
    
    pred_biom = pred_biom.flatten()
    
    # Save the predicted biometry measurements
    if args.output_biom:
        columns = ['LCC', 'HV', 'bBIP', 'sBIP', 'TCD']
        biom_df = pd.DataFrame([pred_biom], columns=columns)
        biom_df.to_csv(args.output_biom, sep='\t', index=False)
        print(f"Biometry values saved to {args.output_biom}")
    
    # The output segmentation is not implemented here; it's just a placeholder.
    if args.output_seg:
        print(f"Segmentation output should be saved to {args.output_seg}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on a single subject")
    parser.add_argument('--input', type=str, required=True, help="Path to the input T2w image in NIfTI format")
    parser.add_argument('--participants', type=str, help="Path to the participants.tsv file")
    parser.add_argument('--output_seg', type=str, help="Path to save the output segmentation NIfTI file")
    parser.add_argument('--output_biom', type=str, help="Path to save the output biometry TSV file")

    args = parser.parse_args()
    main(args)
