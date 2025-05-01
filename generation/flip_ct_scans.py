import os
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes


def flip_nifti_vertically_180(input_path, output_path):
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()
    flipped_data = np.flip(np.flip(data, axis=0), axis=1)
    flipped_img = nib.Nifti1Image(flipped_data, affine=nifti_img.affine, header=nifti_img.header)
    nib.save(flipped_img, output_path)
    print(f"Flipped and saved: {output_path}")


def process_nifti_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            if os.path.exists(output_path):
                print(f"Skipping (already exists): {output_path}")
                continue
            flip_nifti_vertically_180(input_path, output_path)


def check_orientations_in_directory(input_dir, expected_orientation=('R', 'A', 'S')):
    print(f"Checking NIfTI orientations in directory: {input_dir}\n")
    has_differences = False
    for filename in os.listdir(input_dir):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            filepath = os.path.join(input_dir, filename)
            try:
                img = nib.load(filepath)
                orientation = aff2axcodes(img.affine)
                if orientation != expected_orientation:
                    print(f"{filename} ➜ Orientation: {orientation} ❌")
                    has_differences = True
                else:
                    print(f"{filename} ➜ Orientation: {orientation} ✅")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    if not has_differences:
        print("\n All scans match the expected orientation:", expected_orientation)
    else:
        print("\n️ At least one scan does not match the expected orientation.")
