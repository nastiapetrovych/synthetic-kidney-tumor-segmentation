import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def analyse_tumor_data(dataset, source_dir):
    # Ensure boolean columns are correctly interpreted
    dataset["Rtumor"] = dataset["Rtumor"].astype(str).str.lower() == "true"
    dataset["Ltumor"] = dataset["Ltumor"].astype(str).str.lower() == "true"

    # Select only cases with a tumor
    tumor_cases = dataset[(dataset["Rtumor"] == True) | (dataset["Ltumor"] == True)]

    # Select the first tumor case
    selected_case = tumor_cases.iloc[0]

    # Extract file path and statistics
    file_path = selected_case["file_path"]
    # Slice thickness
    z_spacing = selected_case["z_spacing"]
    x_spacing = selected_case["x_spacing"]
    y_spacing = selected_case["y_spacing"]

    # Extract the case ID from the file path
    selected_case_id = file_path.split('tuh_case_')[-1]
    print(f"Selected Case ID: {selected_case_id[:-12]}")
    # Corrected label file name
    label_case_id = f'{selected_case_id[:-12]}.nii.gz'

    # Construct paths for image and corresponding segmentation mask
    image_file = f"{source_dir}/imagesTr/{selected_case_id}"
    mask_file = f"{source_dir}/labelsTr/{label_case_id}"

    ct_img = nib.load(image_file)
    ct_scan = ct_img.get_fdata()

    mask_img = nib.load(mask_file)
    tumor_mask = mask_img.get_fdata()

    if ct_scan.shape != tumor_mask.shape:
        print("Warning: CT scan and mask dimensions do not match!")

    # We select the slice where the tumor has the largest area in the segmentation mask
    slice_tumor_sizes = [np.sum(tumor_mask[:, :, i]) for i in range(tumor_mask.shape[2])]
    # Find the slice with the largest tumor presence
    z_slice = np.argmax(slice_tumor_sizes)

    ct_slice = ct_scan[:, :, z_slice]
    mask_slice = tumor_mask[:, :, z_slice]

    # Convert mask to binary, 1 where tumor exists
    mask_slice = mask_slice > 0

    # Scale axes using spacing values
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(ct_slice, cmap="gray", origin="lower", extent=[0, x_spacing * ct_slice.shape[1], 0, y_spacing * ct_slice.shape[0]])

    # Only plot contours if the mask is not empty
    if np.any(mask_slice):
        ax.contour(mask_slice, colors="red", linewidths=1.5, extent=[0, x_spacing * mask_slice.shape[1], 0, y_spacing * mask_slice.shape[0]])  # Overlay tumor contour
    else:
        print("Warning: No tumor found in the selected slice.")

    ax.set_title(f"CT Scan with Tumor Contour\nBest Slice: {z_slice} | Z-Spacing: {z_spacing} | X-Spacing: {x_spacing} | Y-Spacing: {y_spacing}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    im = ax.imshow(ct_slice, cmap="gray", origin="lower", extent=[0, x_spacing * ct_slice.shape[1], 0, y_spacing * ct_slice.shape[0]])
    fig.colorbar(im, ax=ax, label="Intensity")
    plt.savefig("result_tumor_contour_statistics.png")
    plt.show()
