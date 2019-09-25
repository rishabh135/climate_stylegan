#!/bin/sh
#SBATCH -N 2
#SBATCH -c 20
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00 

module load esslurm
module load cuda/10.0.130
module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07



source activate tf-1.13


pip install tensorflow-datasets
# pip install netCDF4
pip install xarray


#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset FFHQ_128 --img_size 128 --gpu_num 2 --progressive True --phase train


python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/slicing_rbc_data.py
 
	