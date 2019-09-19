#!/bin/sh
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 01:00:00 

module load esslurm
module load cuda/10.0.130
module load python/3.7-anaconda-2019.07

source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36

#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/dataset_prep.py 
 

