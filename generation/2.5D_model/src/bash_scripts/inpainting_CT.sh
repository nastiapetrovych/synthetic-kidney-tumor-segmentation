#!/bin/bash
#SBATCH -o experiments-2.5D/inpainting_CT_kidney_boxed_3slices.log
#SBATCH -J box_kidney_3slices
#SBATCH -A revvity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH -t 40:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000

module load any/python/3.8.3-conda

conda activate /gpfs/space/home/sedykh/.conda/envs/imgx

/gpfs/space/home/sedykh/.conda/envs/imgx/bin/pip install torch

cd /gpfs/space/home/sedykh/BCV/Palette-Image-to-Image-Diffusion-Model

nvidia-smi

/gpfs/space/home/sedykh/.conda/envs/imgx/bin/python3 run.py -c config/inpainting_ct_kidney_boxed_3slices.json -b 20 -gpu 0,1
#a100-80g