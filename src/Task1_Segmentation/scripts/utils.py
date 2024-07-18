import os
import numpy as np
import tensorflow as tf

def create_tf_dataset(data_paths, img_size, num_classes, batch_size, augment=False):
    def data_generator(data_paths, img_size, num_classes, augment=False):
        for img_path, mask_path in data_paths:
            img = load_nifti_image(img_path, img_size)
            mask = load_nifti_label(mask_path, img_size, num_classes)
            if img is None or mask is None:
                continue
            if augment:
                img, mask = augment_image_advanced(img, mask)
            yield img, mask

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data_paths, img_size, num_classes, augment),
        output_signature=(
            tf.TensorSpec(shape=(img_size[0], img_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(img_size[0], img_size[1], num_classes), dtype=tf.float32)
        )
    )
    dataset = dataset.batch(batch_size).repeat()
    return dataset

def save_nifti_image(img_data, reference_image_path, output_path):
    reference_img = nib.load(reference_image_path)
    new_img = nib.Nifti1Image(img_data, affine=reference_img.affine, header=reference_img.header)
    nib.save(new_img, output_path)


