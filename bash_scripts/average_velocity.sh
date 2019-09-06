#!/bin/sh
#SBATCH -N 1
#SBATCH -c 10	
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 04:00:00 

module load esslurm
module load cuda/10.0.130


# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13
pip install xarray
# module load tensorflow/gpu-1.13.1-py36



python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/average_velocity.py