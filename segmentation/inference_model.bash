#!/bin/bash
#SBATCH --job-name="model_inference"
#SBATCH -o logs/model_inference_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --time 60:00:00
#SBATCH --cpus-per-task 16
#SBATCH --mem 96000

module load any/python/3.8.3-conda
conda activate nnunet

export nnUNet_raw="experiments/nnUNet_raw"
export nnUNet_preprocessed="experiments/nnUNet_preprocessed"
export nnUNet_results="experiments/nnUNet_results"

# Run inference of model trained on Dataset d on the 0th fold, with default nnU-Net trainer and 3D full resolution
nnUNetv2_predict -d  612 -i dataset_test_hospital_a/imagesTr -o experiments/results_612_model -f  0 -tr nnUNetTrainer -c 3d_fullres

# Evaluate model on the test set, taking into account Dice score, Intersection over Union, FP, and TP
nnUNetv2_evaluate_folder \
-djfile /experiments/nnUNet_raw/Dataset612/dataset.json \
-pfile /experiments/nnUNet_results/Dataset612/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json \
dataset_test_hospital_a/labelsTr/  /experiments/results_612_model
