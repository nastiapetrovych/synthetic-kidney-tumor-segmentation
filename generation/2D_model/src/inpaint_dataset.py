import logging
import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.transform import resize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_output_dirs(base_dir):
    return {
        "input": os.path.join(base_dir, "input"),
        "gt": os.path.join(base_dir, "gt"),
        "mask": os.path.join(base_dir, "mask"),
        "annotation": os.path.join(base_dir, "annotation"),
    }


BACKGROUND, KIDNEYAREA, TUMOR, CYST = 0, 1, 2, 3
WINDOW_CENTER, WINDOW_WIDTH = 40, 400
TARGET_SHAPE = (512, 512)


def window_and_normalize(image, center, width):
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(image, min_val, max_val)
    return np.interp(windowed, (min_val, max_val), (0, 255)).astype(np.uint8)


def pad_and_resize(image, target_shape):
    padded_size = max(image.shape)
    padded = np.full((padded_size, padded_size), image.min(), dtype=image.dtype)
    y_offset = (padded_size - image.shape[0]) // 2
    x_offset = (padded_size - image.shape[1]) // 2
    padded[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    return resize(padded, target_shape, anti_aliasing=True)


def process_patient_folder(patient_folder, data_dir, output_dirs):
    patient_path = os.path.join(data_dir, patient_folder)
    ct_file = os.path.join(patient_path, "ct.nii.gz")
    seg_file = os.path.join(patient_path, "tumor_1.nii.gz")
    ann_file = os.path.join(patient_path, "annotation_1.nii.gz")

    if not (os.path.exists(ct_file) and os.path.exists(seg_file)):
        return

    try:
        shutil.copy(ann_file, os.path.join(output_dirs["annotation"], f"{patient_folder}.nii.gz"))
        ct_data = nib.load(ct_file).get_fdata()
        seg_data = nib.load(seg_file).get_fdata()

        for slice_idx in range(ct_data.shape[0]):
            slice_img = ct_data[:, :, slice_idx]
            mask = seg_data[:, :, slice_idx] == KIDNEYAREA
            if np.sum(mask) == 0:
                continue

            # Rotate and process slice
            rot_img = np.rot90(slice_img, k=-1)
            rot_mask = np.rot90(mask, k=-1)

            norm_img = window_and_normalize(rot_img, WINDOW_CENTER, WINDOW_WIDTH)
            masked_img = norm_img.copy()
            masked_img[rot_mask] = 255

            input_img = pad_and_resize(masked_img, TARGET_SHAPE)
            gt_img = pad_and_resize(norm_img, TARGET_SHAPE)
            mask_img = pad_and_resize(rot_mask.astype(float), TARGET_SHAPE)

            # Save processed slice
            base_name = f"{patient_folder}_slice_{slice_idx}.png"
            plt.imsave(os.path.join(output_dirs["input"], base_name), input_img, cmap="gray")
            plt.imsave(os.path.join(output_dirs["gt"], base_name), gt_img, cmap="gray")
            plt.imsave(os.path.join(output_dirs["mask"], base_name), mask_img, cmap="gray", vmin=0, vmax=1)
            logging.info(f"Saved: {base_name}")

    except Exception as e:
        logging.error(f"Error processing {patient_folder}: {e}")
