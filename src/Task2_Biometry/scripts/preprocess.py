import os
import numpy as np
import nibabel as nib
from skimage.transform import resize

# Function to load and preprocess images
def load_and_preprocess_image(subject_path, target_shape=(128, 128, 128)):
    t2_file_path = glob.glob(os.path.join(subject_path, 'anat', '*_meas.nii.gz'))

    if not t2_file_path:
        print(f"File not found in path: {subject_path}")
        return None

    t2_file_path = t2_file_path[0]
    t2_img = nib.load(t2_file_path)
    t2_data = t2_img.get_fdata()

    t2_data_resized = resize(t2_data, target_shape, mode='constant', anti_aliasing=True)

    t2_data_norm = (t2_data_resized - np.min(t2_data_resized)) / (np.max(t2_data_resized) - np.min(t2_data_resized))

    return t2_data_norm
