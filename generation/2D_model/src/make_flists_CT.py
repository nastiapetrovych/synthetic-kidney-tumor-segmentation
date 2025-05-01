import os


def make_flist(input_directory, output_dir, filename):
    # Extract unique patient identifiers from the filenames
    all_files = os.listdir(os.path.join(input_directory, "gt"))
    all_identifiers = list(set([file.split('_')[0] for file in all_files]))
    test_identifiers = all_identifiers
    test_files = [file for file in all_files if file.split('_')[0] in test_identifiers]

    with open(f"{output_dir}/{filename}", "w") as f:
        for item in test_files:
            f.write("%s\n" % item)
