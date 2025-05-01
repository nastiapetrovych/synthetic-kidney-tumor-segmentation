#!/bin/bash
#SBATCH --nodes 1
#SBATCH --job-name="2d_model_inference"
#SBATCH -o logs/2d_model_inference_%j.log
#SBATCH --time 60:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task 16
#SBATCH --mem 96000

module load any/python/3.8.3-conda
conda activate nnunet

cd synthetic-kidney-tumor-segmentation/generation/2D_model/src
# Returns CT and resized adjusted tumor mask
python Parnu_with_tumors.py

# Returns 2D images, sliced 3D, where gt is original CT slice, mask is binary mask indicating tumor presence,
# and input is  CT slice with tumor mask overlaid
python inpaint_dataset.py

# Makes flist for inference
python make_flists_CT.py

# Runs inference of the model
cd  synthetic-kidney-tumor-segmentation/generation/Palette-Image-to-Image-Diffusion-Models/
python run.py -p test -c  config/inpainting_ct_kidney.json -b 20
