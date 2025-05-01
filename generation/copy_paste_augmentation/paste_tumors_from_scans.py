import os
import random
import logging
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import feature

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def set_orientation_nib(image):
    orig_ornt = nib.io_orientation(image.affine)
    targ_ornt = nib.orientations.axcodes2ornt("RAS")
    transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)
    return image.as_reoriented(transform)


def load_nifti_data(path):
    img = nib.load(path)
    img = set_orientation_nib(img)
    slice_thickness = img.header['pixdim'][3]
    return img.get_fdata(), img, slice_thickness


def resample_image(image, target_spacing, original_spacing):
    # Resample only along Z-axis
    zoom_factors = [1, 1, original_spacing / target_spacing]
    # Linear interpolation
    return ndimage.zoom(image, zoom_factors, order=1)


def get_tumor_center(segmentation, case_spacing, control_spacing):
    tumor_mask = segmentation == 2
    labeled, num_features = ndimage.label(tumor_mask.astype(int))

    if num_features != 1:
        # Skip ambiguous cases
        return None, None

    center = ndimage.center_of_mass(tumor_mask, labeled, [1])[0]
    tumor_center = list(map(int, center))
    # Adjust Z
    tumor_center[2] = int(round(tumor_center[2] * (case_spacing / control_spacing)))
    return tumor_mask, tumor_center


def pick_random_kidney_edge(kidney_seg):
    single_kidney = kidney_seg.copy()
    left = random.choice([True, False])

    single_kidney[256:, :, :] = 0 if left else single_kidney[256:, :, :]
    single_kidney[:256, :, :] = 0 if not left else single_kidney[:256, :, :]
    kidney_type = "L" if left else "R"

    z_index = np.argmax(np.sum(single_kidney, axis=(0, 1)))
    edge = feature.canny(single_kidney[:, :, z_index], sigma=10)
    edge_coords = list(zip(*np.where(edge)))

    if not edge_coords:
        return None, None, None, None

    edge_coord = random.choice(edge_coords)
    return edge_coord, z_index, single_kidney, kidney_type


def insert_tumor(case_path, control_path, output_path):
    try:
        # Load input images and segmentations
        case_data, _, case_thickness = load_nifti_data(case_path)
        seg_data, _, _ = load_nifti_data(case_path.replace("images", "labels").replace('_0000', ''))

        control_data, control_img, control_thickness = load_nifti_data(control_path)
        kidney_seg, _, _ = load_nifti_data(control_path.replace("images", "labels").replace('_0000', ''))
        kidney_mask = kidney_seg > 0

        if abs(case_thickness - control_thickness) > 0.5:
            logging.info(f"Resampling {control_path} from {control_thickness}mm to {case_thickness}mm")
            control_data = resample_image(control_data, case_thickness, control_thickness)
            kidney_seg = resample_image(kidney_seg, case_thickness, control_thickness)

        tumor_seg, tumor_center = get_tumor_center(seg_data, case_thickness, control_thickness)
        if tumor_center is None:
            logging.warning(f"Skipping {case_path} (ambiguous tumor segmentation).")
            return

        tumor_data = np.zeros_like(tumor_seg, dtype=float)
        tumor_data[tumor_seg] = case_data[tumor_seg]

        edge_coord, z_index, _, kidney_type = pick_random_kidney_edge(kidney_mask)
        if edge_coord is None:
            logging.warning(f"No valid edge for tumor insertion in {control_path}")
            return

        # Compute shifts and roll tumor
        shift = (
            edge_coord[0] - tumor_center[0],
            edge_coord[1] - tumor_center[1],
            z_index - tumor_center[2]
        )

        rolled_tumor = np.roll(np.roll(tumor_data, shift[0], axis=0), shift[1], axis=1)
        temp_volume = np.zeros_like(control_data)

        z_start = max(0, tumor_center[2] - control_data.shape[2] // 2)
        z_end = min(rolled_tumor.shape[2], z_start + control_data.shape[2])
        temp_volume[:, :, :z_end - z_start] = rolled_tumor[:, :, z_start:z_end]
        final_tumor = np.roll(temp_volume, shift[2] + z_start, axis=2)

        # Insert tumor into control
        control_data[final_tumor != 0] = final_tumor[final_tumor != 0]
        kidney_seg[final_tumor != 0] = 2

        nib.save(nib.Nifti1Image(control_data, control_img.affine), output_path)
        nib.save(nib.Nifti1Image(kidney_seg, control_img.affine), output_path.replace("images", "labels").replace(".nii.gz", "_ann.nii.gz"))
        logging.info(f"Saved: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {case_path} with {control_path}: {e}")


def process_all_cases(case_dir, control_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

    case_paths = sorted(os.listdir(case_dir))
    control_paths = sorted(os.listdir(control_dir))

    for case_file in case_paths:
        case_path = os.path.join(case_dir, case_file)
        case_id = os.path.basename(case_path).split("_")[0]

        for control_file in control_paths:
            control_path = os.path.join(control_dir, control_file)
            control_id = os.path.basename(control_file).split("_")[0]
            out_path = os.path.join(output_dir, f"images/{case_id}__{control_id}.nii.gz")

            insert_tumor(case_path, control_path, out_path)
