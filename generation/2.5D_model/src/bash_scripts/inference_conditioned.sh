#!/bin/bash
#SBATCH -o cond_inf.log
#SBATCH -J cond_inf
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 80:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G 
#SBATCH --exclude=falcon3,falcon2

module load any/python/3.8.3-conda
conda activate /gpfs/space/home/sedykh/.conda/envs/imgx

#/gpfs/space/home/sedykh/.conda/envs/imgx/bin/pip install torch

cd /gpfs/space/home/sedykh/BCV/Palette-Image-to-Image-Diffusion-Models

/gpfs/space/home/sedykh/.conda/envs/imgx/bin/python3 inference_conditioned.py   

#a100-80g