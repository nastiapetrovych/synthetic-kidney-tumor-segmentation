#!/bin/bash
#SBATCH -o experiments/inpainting_CT_tumor_kits_windowed.log
#SBATCH -J tumor_inpainting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:4
#SBATCH -t 120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --exclude=falcon3,falcon2

module load any/python/3.8.3-conda
conda activate nnunet

cd Palette-Image-to-Image-Diffusion-Models

python run.py -c config/inpainting_tumor_kits.json -gpu 0,1,2,3 -b 4
