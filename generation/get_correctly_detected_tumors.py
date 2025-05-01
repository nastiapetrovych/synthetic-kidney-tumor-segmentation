import json
import os


def get_correctly_detected_tumors(result_file, tumor_file, result_dir, model_name):
    with open(result_file, 'r') as f:
        results_data = json.load(f)['metric_per_case']

    with open(tumor_file, 'r') as f:
        tumor_scan_files = set(f["tumor_scans"])

    correctly_detected = {}
    false_positives = {}

    for result in results_data:
        file_name = result['reference_file'].replace('.nii.gz', '_0000.nii.gz')
        # Get class 2 (tumors) metrics
        metrics = result['metrics'].get('2', {})

        if not metrics:
            continue

        is_tumor_case = any(file_name.endswith(tumor_file) for tumor_file in tumor_scan_files)

        if is_tumor_case:
            if metrics.get('Dice', 0) >= 0.5 and metrics.get('IoU', 0) >= 0.5:
                correctly_detected[file_name] = metrics
        else:
            if metrics.get('FP', 0) > 1:
                false_positives[file_name] = metrics

    # Output paths
    with open(os.path.join(result_dir, f'metrics_model_{model_name}.json'), 'w') as f:
        json.dump(correctly_detected, f, indent=4)

    with open(os.path.join(result_dir, f'metrics_fp_model_{model_name}.json'), 'w') as f:
        json.dump(false_positives, f, indent=4)
