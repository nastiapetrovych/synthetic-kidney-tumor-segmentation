import os
import numpy as np
import nibabel as nib
from PIL import Image


def convert_pngs_to_nifti(
    input_directory,
    output_directory,
    base_filename,
    output_filename_middle_channel,
    output_filename_all_channels,
    save_all_channels,
    slice_step,
    selected_channel
):
    file_list = sorted(
        [img for img in os.listdir(input_directory) if img.startswith(base_filename)],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    print(f"Found {len(file_list)} matching image slices.")

    # Load images
    image_slices = [Image.open(os.path.join(input_directory, img)) for img in file_list]
    # (num_slices, H, W[, C])
    image_data = np.stack([np.array(img) for img in image_slices])
    print("Image data shape:", image_data.shape)

    affine = np.eye(4)

    # Extract selected channels and transpose to (H, W, D)
    if image_data.ndim == 4:
        middle_channel_slices = image_data[:, :, :, selected_channel]
    else:
        middle_channel_slices = image_data

    middle_channel_slices = np.transpose(middle_channel_slices, (1, 2, 0))
    output_path_middle = os.path.join(output_directory, output_filename_middle_channel)
    nib.save(nib.Nifti1Image(middle_channel_slices, affine), output_path_middle)
    print(f"Saved middle channel to: {output_path_middle}")

    # Optionally save all channels stacked (flattened)
    if save_all_channels and image_data.ndim == 4:
        selected_slices = image_data[::slice_step]
        # (H, W, D, C)
        selected_slices = np.transpose(selected_slices, (1, 2, 0, 3))
        flattened = selected_slices.reshape(selected_slices.shape[0], selected_slices.shape[1], -1)
        output_path_all = os.path.join(output_directory, output_filename_all_channels)
        nib.save(nib.Nifti1Image(flattened, affine), output_path_all)
        print(f"Saved all channels stacked to: {output_path_all}")
