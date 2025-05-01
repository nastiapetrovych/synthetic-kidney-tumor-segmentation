import nibabel as nib
import numpy as np
import imageio.v3 as iio
import os
from skimage.transform import resize
from PIL import Image
from torchvision import transforms
from collections import defaultdict


def normalize_and_scale_image(image, window_center, window_width):
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed_image = np.clip(image, lower, upper)
    normalized_image = np.interp(windowed_image, (lower, upper), (0, 255))
    return normalized_image.astype(np.uint8)


def custom_transform(image):
    image = image.copy()
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize(mean=[0.5], std=[0.5])(tensor)
    return tensor


def transform_entire_volume_to_neg1pos1(ct_data_3d, window_center, window_width):
    h, w, z = ct_data_3d.shape
    out_vol = np.zeros_like(ct_data_3d, dtype=np.float32)
    for slice_idx in range(z):
        slice_2d = ct_data_3d[:, :, slice_idx]
        slice_norm = normalize_and_scale_image(slice_2d, window_center, window_width)
        slice_tensor = custom_transform(Image.fromarray(slice_norm))
        slice_neg1_pos1 = slice_tensor.squeeze(0).numpy()
        out_vol[:, :, slice_idx] = slice_neg1_pos1.astype(np.float32)
    return out_vol


def main(model_output, ct_scans_dir, output_dir):
    window_center = 40
    window_width = 400

    # Group PNG files by file ID
    png_files = [f for f in os.listdir(model_output) if f.startswith("Out_2.25") and f.endswith(".png")]
    file_id_to_slices = defaultdict(list)

    for f in png_files:
        parts = f.split("slice_")
        file_id = parts[0].replace("Out_", "").rstrip("_")
        slice_idx = int(parts[1].replace(".png", ""))
        file_id_to_slices[file_id].append((slice_idx, f))
    print(len(file_id_to_slices.keys()))

    for file_id, slices in file_id_to_slices.items():
        nifti_path = os.path.join(ct_scans_dir, f"{file_id}_0000.nii.gz")
        if not os.path.exists(nifti_path):
            print(f"Skipping {file_id} because NIfTI file not found.")
            continue

        print(f"\nProcessing file ID: {file_id}")
        original_nifti = nib.load(nifti_path)
        original_data = original_nifti.get_fdata()
        affine = original_nifti.affine
        header = original_nifti.header
        target_shape = original_data.shape[:2]

        ct_scan_data = original_data.copy()
        ct_scan_data = transform_entire_volume_to_neg1pos1(ct_scan_data, window_center, window_width)

        for slice_idx, png_file in slices:
            print(f" - Processing slice {slice_idx}")
            png_slice = iio.imread(os.path.join(model_output, png_file), mode="L")
            resized_png_slice = resize(np.array(png_slice), target_shape, anti_aliasing=True)
            if resized_png_slice.shape != original_data[:, :, 0].shape:
                raise ValueError(f"Shape mismatch: PNG {resized_png_slice.shape} vs NIfTI {original_data[:, :, 0].shape}")
            resized_png_slice = np.rot90(resized_png_slice, k=1)
            resized_slice_tensor = custom_transform(resized_png_slice)
            if slice_idx < original_data.shape[-1]:
                ct_scan_data[:, :, slice_idx] = resized_slice_tensor

        transformed_ct_scan = np.rot90(ct_scan_data, k=2)
        output_path = os.path.join(output_dir, f"{file_id}_0000.nii.gz")
        nib.save(nib.Nifti1Image(transformed_ct_scan, affine, header), output_path)
        print(f"Saved new NIfTI for {file_id} to {output_path}")


model_output = "/gpfs/space/projects/BetterMedicine/nastiap/2D_model/Palette-Image-to-Image-Diffusion-Models/experiments/test_inpainting_ct_kidney_parnu_250424_050801/results/test/0"
nifti_dir = "/gpfs/space/projects/BetterMedicine/nastiap/2D_model/clean_scans_dir/images/"
output_dir = "/gpfs/space/projects/BetterMedicine/nastiap/2D_model/generated_scans_200/images"
main(model_output, nifti_dir, output_dir)
