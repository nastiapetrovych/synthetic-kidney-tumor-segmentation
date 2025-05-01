# Evaluating the Impact of Synthetic Data on Enhancing Kidney Tumor Segmentation in CT Scans

This project investigates how different synthetic data generation techniques influence the performance of medical image segmentation models, with a specific focus on kidney tumors in CT scans. The primary model used in this study is nnU-Net, a state-of-the-art framework for biomedical image segmentation.

This research was conducted as part of the undergraduate thesis of Anastasiia Petrovych, under the supervision of Dmytro Fishman.



## 🧠 Thesis Goal
To explore whether injecting synthetic kidney tumors into real CT scans can improve the accuracy and flexibility of segmentation models. The research examines:

* The effectiveness of various synthetic tumor generation methods
  
* Performance changes in nnU-Net models trained on augmented datasets
  
* Visual and quantitative evaluation of segmentation results



> Note: We cannot provide access to complete dataset due to healthcare data privacy restrictions, therefore only one .nii.gz file is included as an example.

If further details are required, please contact me at anastasiia.petrovych@ucu.edu.ua.


## 📁 Project Structure


```
.
├── data/                            # Metadata and filtered dataset information
│   ├── filtered_hospital_a_tumors.csv
│   ├── hospital_a_ct_scans_with_tumors.json
│   └── hospital_b_ct_scans_with_tumors.json

├── experiments/                     # nnU-Net preprocessed inputs, raw data, and training results
│   ├── nnUNet_preprocessed/
│   ├── nnUNet_raw/
│   └── nnUNet_results/

├── generation/                      # Synthetic data generation and analysis tools
│   ├── 2D_model/                    # Diffusion-based 2D tumor generation model
│   ├── 2.5D_model/                  # Multi-slice (2.5D) generation model
│   ├── Palette-Image-to-Image-Diffusion-Models/  # Palette diffusion framework
│   ├── copy_paste_augmentation/     # Copy-paste based tumor injection
│   ├── tumor_masks/                 # Extracted or generated tumor masks
│   ├── analyse_tumor_data.py        # Analyse tumor distribution and sizes
│   ├── augment_dataset.py           # Apply augmentations to CT datasets
│   ├── flip_ct_scans.py             # Flip scans to RAS orientation
│   ├── check_file_corrupted.py      # Check for corrupted CT or mask files
│   └── get_correctly_detected_tumors.py  # Evaluate tumor detection performance

├── segmentation/                    # nnU-Net training and inference scripts
│   ├── nnUNet/                      # nnU-Net framework
│   ├── train_model.bash             # Training model script
│   └── inference_model.bash         # Inference and evaluation script

├── visualisation/                   # Plots, overlays, and result visualizations

├── requirements.txt                 # List of Python dependencies
└── README.md                        # Project documentation
```


## How to start

> Note: To run the majority of scripts in this repository (including model training and inference), high-computing resources with NVIDIA A100 GPU. In our case, we use a high-performance computing (HPC) cluster of University of Tartu.

1. Clone this repository to local environemnt.

2. Install Python requirements.
```bash
pip install requirements.txt
```

3. Train the model.

Send SLURM job.
```bash
sbatch segmentation/train_model.bash
```

Wait until the model is trained. The example of my pre-trained checkpoint [Dataset607 best checkpoint](https://drive.google.com/file/d/1h4NlAF55b0y9a3DNZLXnsE80d-qGWw8e/view?usp=sharing)

4. Run inference to get results.

Send SLURM job.
```bash
sbatch segmentation/inference_model.bash
```


### If you want to try generations:

Go to `generation/` folder.

If you want to run 2D model -> navigate to `2D_model/`.

If you want to run 2.5D model -> navigate to `2.5D_model/`.

If you want to run Copy-paste augmentation -> navigate to `copy_paste_augmentation/`.


### Files on Google Drive

Model checkpoint and dataset sample files exceed the limit of GitHub files (100MB), so they were uploaded to Google Drive.
Download the following files by this link: https://drive.google.com/drive/folders/1ZoGpWzeFLITlxsb92CdHcDxVw3Q5L91p 



## Solution

The proposed approach includes a systematic comparison between copy-paste augmentation and generative models, with a particular focus on diffusion-based methods. To address the problem of the limited amount of annotated data, we adopt multiple strategies: (1) optimal training-validation-test data splitting, (2) fine-tuning of pretrained models on new datasets, (3) copy-paste augmentation, (4) synthetic data generation using 2D diffusion models, and (5) synthetic data generation using 2.5D diffusion models. The results of every approach are validated, using standard segmentation metrics, on a held-out test set to ensure fair comparison and reproducibility.

The proposed pipeline, illustrated on Figure 1, provides an overview of all major components.

![Pipeline diagram](diagram_pipeline.png)
*Figure 1. Overview of the proposed pipeline. Real clinical data from Estonian hospitals A and B is used.*


### Synthetic Data Generation Approaches

This section outlines three ways to create realistic tumor-bearing CT scans from clean images. The simplest method, copy-paste augmentation, takes real 3D tumor patches (including a bit of surrounding tissue), aligns and resamples them to match a clean scan, then pastes them into a randomly chosen kidney region—using edge detection to find a natural spot—and updates the scan’s pixel values and mask. The second method uses a 2D diffusion model (a fine-tuned DDPM) that “inpaints” tumors onto individual CT slices by filling in user-provided masks with textures and shapes learned from real tumors. Finally, the 2.5D diffusion model improves on this by feeding three consecutive slices (only one of which is masked) through the network in sequence, so the generated tumors stay coherent across adjacent slices; the newly created regions are then stitched back into the full volume for smooth, anatomically consistent results. Examples of outputs can be seen on Figure 2.

![Tumors](tumors.png)
*Figure 2. Comparison of original CT scans and synthetic tumor generation results. The top row presents the original scan with a tumor, the copy-paste augmentation output, the 2D diffusion model output, and the 2.5D diffusion model output. The bottom row shows the same scans with tumor regions highlighted in red for clarity.*


### Segmentation Model Performance

We evaluated kidney tumor segmentation using nnU-Net across four datasets: KiTS, KIRC, and two clinical collections (Datasets A and B). Our baseline model—trained on KiTS, KIRC, and Dataset A—achieved strong results on Dataset A, DSC **0.86**, IoU **0.81**, but failed to generalise to Dataset B, DSC **0.07**, IoU **0.06**, highlighting domain shifts and class imbalance. Fine-tuning on Dataset B alone proved ineffective due to its scarcity of tumors, even when merged with source data. To address this, we introduced synthetic scans via copy-paste augmentation, 2D diffusion, and 2.5D diffusion models. Copy-paste augmentation yielded modest gains, DSC **0.10**, IoU **0.09** on Split II, whereas diffusion-based methods underperformed. Finally, combining source data with Dataset B plus copy-paste-generated scans improved detection, DSC **0.09**, IoU **0.08**, four out of six tumors detected, confirming that simple synthetic augmentation can partially mitigate distribution gaps and class imbalance.

![Segmentation masks](segmentation_masks_upd.png)
*Figure 3. Visual examples of segmentation results on CT slices of Dataset B. Green contours represent the ground truth tumor masks, and pink contours represent the model’s predictions. The pink area indicates a false positive if no green contour is present. DSC and IoU scores are provided for each case.*


## Summary of Results

The experimental results demonstrate that augmentation with synthetic data positively impacts model performance, providing slight improvements compared to other strategies. Neither optimal training-validation-test data splitting strategies nor fine-tuning alone leads to significant improvements across different settings. In contrast, models trained with synthetic data augmentation show better performance across different dataset configurations.
Although the observed improvements are minor, they demonstrate the potential of synthetic data augmentation, even with current limitations in the methods used. This suggests promising directions for future research. A detailed overview of all experimental results is provided in Table 1.


---

**Test Set Dataset A:** 192 CT Scans (99 with tumor lesions, 93 without tumors)

| Models                                                   | DSC ↑          | IoU ↑          | TP ↑           | FP ↓  |
|----------------------------------------------------------|----------------|----------------|----------------|-------|
| <u>Source-Only</u>                                       | <u>0.86</u>    | <u>0.81</u>    | <u>0.95</u>    | 0.05  |
| Target-Only                                              | 0.00           | 0.00           | 0.00           | 0.00  |
| Pretrained on Source + Finetuned on Target               | 0.00           | 0.00           | 0.00           | 0.00  |
| Source + Target (Joint)                                  | 0.82           | 0.77           | 0.92           | 0.09  |

---

**Test Set Dataset B Split I:** 143 CT Scans (15 with tumor lesions, 128 without tumors)

| Models                                                   | DSC ↑          | IoU ↑          | TP ↑           | FP ↓  |
|----------------------------------------------------------|----------------|----------------|----------------|-------|
| <u>Source-Only</u>                                       | <u>0.07</u>    | <u>0.06</u>    | <u>0.40</u>    | 0.63  |
| Target-Only                                              | 0.00           | 0.00           | 0.00           | 0.00  |
| Pretrained on Source + Finetuned on Target               | 0.00           | 0.00           | 0.00           | 0.00  |

---

**Test Set Dataset B Split II:** 100 CT Scans (6 with tumor lesions, 94 without tumors)

| Models                                                             | DSC ↑       | IoU ↑       | TP ↑        | FP ↓  |
|--------------------------------------------------------------------|-------------|-------------|-------------|-------|
| Source-Only                                                        | 0.06        | 0.06        | 0.67        | 0.62  |
| Target-Only                                                        | 0.00        | 0.00        | 0.00        | 0.00  |
| Pretrained on Source + Finetuned on Target                         | 0.00        | 0.00        | 0.00        | 0.00  |
| Source + Target (Joint)                                            | 0.07        | 0.06        | 0.67        | 0.51  |
| <u>Target-Only with Copy-Paste Augmentation</u>                    | <u>0.10</u> | <u>0.09</u> | 0.17        | 0.03  |
| Target-Only with 2D Diffusion Model Augmentation                   | 0.05        | 0.04        | 0.00        | 0.07  |
| Target-Only with 2.5D Diffusion Model Augmentation                 | 0.004       | 0.002       | 0.00        | 0.04  |
| <u>Source + Target (Joint) with Copy-Paste Augmentation</u>       | 0.09        | 0.08        | 0.67        | 0.37  |
| Source + Target (Joint) with 2.5D Diffusion Model Augmentation     | 0.08        | 0.08        | 0.67        | 0.44  |

---

*Table 1: Comparison of segmentation models’ results across different training datasets, strategies, and data augmentation methods.*
