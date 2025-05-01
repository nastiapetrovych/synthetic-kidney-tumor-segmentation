#!/bin/bash
#SBATCH -o cond_inf.log
#SBATCH -J cond_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH -t 80:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G 
#SBATCH --exclude=falcon3,falcon2

module load any/python/3.8.3-conda
conda activate nnunet


cd Palette-Image-to-Image-Diffusion-Models
python inference_conditioned.py
