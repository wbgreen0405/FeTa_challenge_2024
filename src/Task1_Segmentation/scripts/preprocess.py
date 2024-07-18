import nibabel as nib
import cv2
import numpy as np

def load_nifti_image(img_path, img_size):
    try:
        img = nib.load(img_path).get_fdata().astype(np.float32)
        mid_slice = img.shape[2] // 2
        img = img[:, :, mid_slice]
        img = cv2.resize(img, img_size)
        img = np.stack((img,) * 3, axis=-1)  # Convert to 3 channels

        # Normalize dynamically based on the min and max values of the image
        img_min = np.min(img)
        img_max = np.max(img)
        img = (img - img_min) / (img_max - img_min + 1e-8)  # Adding epsilon to avoid division by zero

        return img
    except Exception as e:
        print(f"Error loading image file {img_path}: {e}")
        return None

def load_nifti_label(mask_path, img_size, num_classes, problematic_files):
    try:
        if mask_path in problematic_files:
            raise Exception("Problematic file")
        mask = nib.load(mask_path).get_fdata().astype(np.float32)
        mid_slice = mask.shape[2] // 2
        mask = mask[:, :, mid_slice]
        mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
        mask = tf.keras.utils.to_categorical(mask, num_classes=num_classes)
        return mask
    except Exception as e:
        print(f"Error loading mask file {mask_path}: {e}")
        return None
