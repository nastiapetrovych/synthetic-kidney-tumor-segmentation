import os
import numpy as np
import nibabel as nib
from imageio import imread


def convert_png_stack_to_nifti(png_dir, output_path, filename_filter):
    png_files = sorted([f for f in os.listdir(png_dir) if f.endswith('.png') and filename_filter in f])

    if not png_files:
        raise FileNotFoundError(f"No PNG files with filter '{filename_filter}' found in: {png_dir}")

    image_data = [imread(os.path.join(png_dir, file)) for file in png_files]
    image_3d = np.stack(image_data, axis=-1)

    nifti_img = nib.Nifti1Image(image_3d, affine=np.eye(4))
    nib.save(nifti_img, output_path)
    print(f"Saved PNG stack as NIfTI: {output_path}")


def convert_npy_stack_to_nifti(npy_dir, output_path, filename_filter):
    npy_files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy') and filename_filter in f])

    if not npy_files:
        raise FileNotFoundError(f"No NPY files with filter '{filename_filter}' found in: {npy_dir}")

    image_data = []
    for file in npy_files:
        file_path = os.path.join(npy_dir, file)
        img = np.load(file_path)
        img = np.squeeze(img)
        image_data.append(img)

    print(f"Example shape of slice: {image_data[0].shape}")
    image_3d = np.stack(image_data, axis=-1)

    nifti_img = nib.Nifti1Image(image_3d, affine=np.eye(4))
    nib.save(nifti_img, output_path)
    print(f"Saved NPY stack as NIfTI: {output_path}")
