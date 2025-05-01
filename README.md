# Evaluating the Impact of Synthetic Data on Enhancing Kidney Tumor Segmentation in CT Scans

This project investigates how different synthetic data generation techniques influence the performance of medical image segmentation models, with a specific focus on kidney tumors in CT scans. The primary model used in this study is nnU-Net, a state-of-the-art framework for biomedical image segmentation.

This research was conducted as part of the undergraduate thesis of Anastasiia Petrovych, under the supervision of Dmytro Fishman.



## 🧠 Thesis Goal
To explore whether injecting synthetic kidney tumors into real CT scans can improve the accuracy and flexibility of segmentation models. The research examines:

  The effectiveness of various synthetic tumor generation methods
  
  Performance changes in nnU-Net models trained on augmented datasets
  
  Visual and quantitative evaluation of segmentation results



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

├── metrics/                         # Quantitative results and evaluation metrics
│   ├── dataset_a/
│   ├── dataset_b_split_I/
│   └── dataset_b_split_II/

├── models/                          # Trained model checkpoints and weights

├── segmentation/                    # nnU-Net training and inference scripts
│   ├── nnUNet/                      # nnU-Net framework
│   ├── train_model.bash             # Training model script
│   └── inference_model.bash         # Inference and evaluation script

├── visualisation/                   # Plots, overlays, and result visualizations

├── requirements.txt                 # List of Python dependencies
└── README.md                        # Project documentation
```
