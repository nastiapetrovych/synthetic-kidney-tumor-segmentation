import os
import SimpleITK as sitk
import numpy as np


def is_orthonormal(direction, tol=1e-6):
    direction_matrix = np.array(direction).reshape(3, 3)
    # Checks if columns are unit vectors, norm == 1
    norms = np.linalg.norm(direction_matrix, axis=0)
    is_unit_vectors = np.allclose(norms, 1, atol=tol)

    # Checks if the matrix is orthogonal, dot product of columns == identity
    orthogonality = np.allclose(
        np.dot(direction_matrix.T, direction_matrix), np.eye(3), atol=tol
    )

    return is_unit_vectors and orthogonality


def check_nifti_file(file_path):
    try:
        # Attempts to read the NIfTI file
        print(file_path)
        img = sitk.ReadImage(file_path)
        direction = img.GetDirection()
        if not is_orthonormal(direction):
            return False
        return True
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def find_corrupted_files(input_dir):
    corrupted_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                if not check_nifti_file(file_path):
                    corrupted_files.append(file_path)
    return corrupted_files
