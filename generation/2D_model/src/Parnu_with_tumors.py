import os
import random
import logging
import numpy as np
import nibabel as nib
import tqdm
from scipy import ndimage
import torchio as tio

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def calculate_volume(mask):
    return np.sum(mask > 0)


def smooth_mask(mask, sigma=1.0):
    return (ndimage.gaussian_filter(mask.astype(float), sigma=sigma) > 0.5).astype(np.uint8)


def make_affine_positive(image):
    affine = image.affine.copy()
    spacing = np.abs(np.diag(affine)[:3])
    signs = np.sign(np.diag(affine)[:3])
    for i in range(3):
        if signs[i] < 0:
            affine[:3, i] *= -1
    return nib.Nifti1Image(image.get_fdata(), affine, image.header)


def resample_image(image):
    spacing = np.abs(np.diag(image.affine)[:3])
    subject = tio.Subject(ct=tio.ScalarImage(tensor=image.get_fdata()[None], affine=image.affine))
    resampled = tio.Resample(spacing)(subject)
    return resampled['ct'].numpy().squeeze(), resampled['ct'].affine


def split_kidneys(mask):
    mid = mask.shape[1] // 2
    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    left[:, :mid, :] = mask[:, :mid, :]
    right[:, mid:, :] = mask[:, mid:, :]
    return left, right


def adjust_scaling_to_fit(tumor_shape, kidney_shape):
    return min(*(k / t for k, t in zip(kidney_shape, tumor_shape)), 1.0)


def create_augmentations(scale, degrees=15):
    return tio.Compose([
        tio.RandomAffine(scales=(scale, scale), degrees=(degrees, degrees, degrees), isotropic=True,
                         image_interpolation='linear')
    ])


def augment_and_resample_tumor(kidney_shape, ct_image, tumor_image):
    tumor_data = tumor_image.get_fdata()
    tumor_shape = tumor_data.shape
    scale = adjust_scaling_to_fit(tumor_shape, kidney_shape)
    scale = min(scale, 2.0)
    ct_spacing = np.abs(np.diag(ct_image.affine)[:3])
    tumor_subject = tio.Subject(tumor=tio.LabelMap(tensor=tumor_data[None], affine=tumor_image.affine))
    resampled = tio.Resample(ct_spacing, image_interpolation='nearest')(tumor_subject)
    try:
        return create_augmentations(scale)(resampled)['tumor'].numpy().squeeze()
    except Exception as e:
        logging.error(f"Augmentation error: {e}")
        return None


def place_tumor(tumor_data, ct_data, voxel):
    if np.count_nonzero(tumor_data) == 0:
        logging.warning("Tumor mask is empty.")
        return None
    tumor_coords = np.argwhere(tumor_data > 0)
    z_min, y_min, x_min = tumor_coords.min(axis=0)
    z_max, y_max, x_max = tumor_coords.max(axis=0) + 1
    tumor_crop = tumor_data[z_min:z_max, y_min:y_max, x_min:x_max]
    z0, y0, x0 = (max(0, v - s // 2) for v, s in zip(voxel, tumor_crop.shape))
    z1, y1, x1 = (min(d, s + o) for d, s, o in zip(ct_data.shape, tumor_crop.shape, (z0, y0, x0)))
    mask = np.zeros_like(ct_data, dtype=np.uint8)
    mask[z0:z1, y0:y1, x0:x1] = (tumor_crop[:z1 - z0, :y1 - y0, :x1 - x0] > 0).astype(np.uint8)
    return mask


def integrate_tumor(ct_data, kidney_mask, tumor_data):
    boundary = np.logical_xor(ndimage.binary_dilation(kidney_mask == 1), kidney_mask == 1)
    voxels = np.argwhere(boundary)
    if voxels.size == 0:
        voxels = np.argwhere(kidney_mask == 1)
    if voxels.size == 0:
        return None
    voxel = voxels[random.randint(0, len(voxels) - 1)]
    return place_tumor(tumor_data, ct_data, voxel)


def select_kidney(vol_left, vol_right, min_vol=10000):
    options = []
    if vol_left > min_vol:
        options.append("left")
    if vol_right > min_vol:
        options.append("right")
    if len(options) == 2:
        return random.choice(["left", "right", "both"])
    return options[0] if options else None


def parnu_with_tumors(images_dir, labels_dir, tumor_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in tqdm.tqdm(os.listdir(images_dir)):
        if not file.endswith('_0000.nii.gz'):
            continue
        subject_id = file.replace('_0000.nii.gz', '')
        ct_path = os.path.join(images_dir, file)
        label_path = os.path.join(labels_dir, f'{subject_id}.nii.gz')
        subj_dir = os.path.join(output_dir, subject_id)
        if os.path.exists(os.path.join(subj_dir, 'tumor_1.nii.gz')):
            logging.info(f"Skipping {subject_id}, already processed.")
            continue
        os.makedirs(subj_dir, exist_ok=True)
        ct_img = make_affine_positive(nib.load(ct_path))
        resampled_data, resampled_affine = resample_image(ct_img)
        ct_img = nib.Nifti1Image(resampled_data, resampled_affine)
        kidney_data = nib.load(label_path).get_fdata()
        left, right = split_kidneys(kidney_data)
        vol_left, vol_right = calculate_volume(left), calculate_volume(right)
        choice = select_kidney(vol_left, vol_right)
        if not choice:
            logging.info(f"{subject_id}: insufficient kidney volume.")
            continue
        logging.info(f"{subject_id}: injecting into {choice} kidney(s)")
        for var in range(1, 3):
            for side in (["left", "right"] if choice == "both" else [choice]):
                tumor_file = random.choice(os.listdir(tumor_dir))
                tumor_img = nib.load(os.path.join(tumor_dir, tumor_file))
                kidney = left if side == "left" else right
                tumor_data = augment_and_resample_tumor(kidney.shape, ct_img, tumor_img)
                if tumor_data is None:
                    continue
                mask = integrate_tumor(resampled_data, kidney, tumor_data)
                if mask is None:
                    continue
                nib.save(nib.Nifti1Image(resampled_data, resampled_affine), os.path.join(subj_dir, 'ct.nii.gz'))
                nib.save(nib.Nifti1Image(mask, resampled_affine), os.path.join(subj_dir, f'tumor_{var}.nii.gz'))
                full_kidney_mask = tio.Resample(tio.ScalarImage(tensor=resampled_data[None], affine=resampled_affine))(
                    tio.Subject(mask=tio.LabelMap(tensor=kidney_data[None], affine=ct_img.affine)))[
                    'mask'].numpy().squeeze().astype(np.uint8)
                full_kidney_mask[mask == 1] = 2
                nib.save(nib.Nifti1Image(full_kidney_mask, resampled_affine),
                         os.path.join(subj_dir, f'annotation_{var}.nii.gz'))
                logging.info(f"Saved subject {subject_id}, variant {var}, kidney={side}")
