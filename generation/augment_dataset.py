import shutil
from pathlib import Path


def extract_base_name(path: Path) -> str:
    return path.stem.replace("_0000", "")


def copy_synthetic_samples(real_image_dir, real_label_dir, generated_image_dir, generated_label_dir, real_scan_count,
                           synthetic_ratio):
    real_image_dir = Path(real_image_dir)
    real_label_dir = Path(real_label_dir)
    generated_image_dir = Path(generated_image_dir)
    generated_label_dir = Path(generated_label_dir)

    # Build dictionaries using the base name (without _0000)
    generated_images = {extract_base_name(p): p for p in generated_image_dir.glob("*.nii*")}
    generated_labels = {extract_base_name(p): p for p in generated_label_dir.glob("*.nii*")}

    # Find matching basenames
    common_basenames = sorted(set(generated_images.keys()) & set(generated_labels.keys()))

    num_to_copy = int(real_scan_count * synthetic_ratio)
    print(f"Preparing to copy {min(num_to_copy, len(common_basenames))} synthetic scan-label pairs.")

    for base_name in common_basenames[:num_to_copy]:
        image_src = generated_images[base_name]
        label_src = generated_labels[base_name]

        image_dst = real_image_dir / image_src.name
        label_dst = real_label_dir / label_src.name

        shutil.copy(image_src, image_dst)
        shutil.copy(label_src, label_dst)

        print(f"Copied image: {image_src.name} -> {image_dst}")
        print(f"Copied label: {label_src.name} -> {label_dst}")
