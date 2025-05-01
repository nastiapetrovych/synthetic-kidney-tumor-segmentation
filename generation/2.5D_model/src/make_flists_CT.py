import os
import random


def split_dataset_by_identifier(data_root, gt_subfolder, train_ratio, train_flist_path, test_flist_path):
    gt_dir = os.path.join(data_root, gt_subfolder)
    all_files = os.listdir(gt_dir)

    # Extract unique patient identifiers (based on first part of filename)
    identifiers = list(set([f.split('_')[0] for f in all_files]))
    random.shuffle(identifiers)

    num_train = int(len(identifiers) * train_ratio)
    train_ids = set(identifiers[:num_train])
    test_ids = set(identifiers[num_train:])

    # Filter files by identifiers
    train_files = [f for f in all_files if f.split('_')[0] in train_ids]
    test_files = [f for f in all_files if f.split('_')[0] in test_ids]

    os.makedirs(os.path.dirname(train_flist_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_flist_path), exist_ok=True)

    with open(train_flist_path, "w") as f:
        for file in train_files:
            f.write(f"{file}\n")

    with open(test_flist_path, "w") as f:
        for file in test_files:
            f.write(f"{file}\n")

    print(f"Split complete: {len(train_files)} train, {len(test_files)} test files.")


def generate_flist_from_directory(input_dir, output_flist_path):
    input_files = os.listdir(input_dir)
    os.makedirs(os.path.dirname(output_flist_path), exist_ok=True)

    with open(output_flist_path, "w") as f:
        for file in input_files:
            f.write(f"{file}\n")

    print(f"Flist created with {len(input_files)} files at: {output_flist_path}")
