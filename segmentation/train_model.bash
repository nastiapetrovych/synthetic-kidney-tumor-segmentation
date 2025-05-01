#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name="model_train"
#SBATCH -o logs/model_train_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --time 120:00:00
#SBATCH --cpus-per-task 8
#SBATCH --mem 128000

module load any/python/3.8.3-conda
conda activate nnunet

export nnUNet_raw="/experiments/nnUNet_raw"
export nnUNet_preprocessed="/experiments/nnUNet_preprocessed"
export nnUNet_results="/experiments/nnUNet_results"

# Data preprocessing of Dataset d, on 3D full resolution, using 8 workers
nnUNetv2_plan_and_preprocess -d 612 -c 3d_fullres --verify_dataset_integrity -np 8
# Model training on Dataset d, 3D full resolution and 0th fold
nnUNetv2_train 612 3d_fullres 0