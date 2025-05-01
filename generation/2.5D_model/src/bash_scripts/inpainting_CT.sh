#!/bin/bash
#SBATCH --job-name="box_kidney_3slices"
#SBATCH -o logs/inpainting_CT_kidney_boxed_3slices_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH -t 40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000

module load any/python/3.8.3-conda
conda activate nnunet

cd Palette-Image-to-Image-Diffusion-Model

python run.py -c config/inpainting_ct_kidney_boxed_3slices.json -b 20 -gpu 0,1