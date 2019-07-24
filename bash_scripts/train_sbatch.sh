#!/bin/sh

#SBATCH -t 04:00:00 
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:8
#the minimum amount of time the job should run


module load cuda
module load nccl

module load esslurm
module load nccl
module load python3
module load tensorflow/gpu-1.13.0-py36


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset FFHQ_128 --img_size 128 --gpu_num 8 --progressive True --phase train

