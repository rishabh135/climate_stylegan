#!/bin/sh
#SBATCH -N 1
#SBATCH -c 40
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:8
#SBATCH -t 08:00:00 

module load esslurm
module load cuda/10.0.130
module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07

source activate tf-1.13


module load tensorflow/gpu-1.13.1-py36
pip install tensorflow-datasets



#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset FFHQ_128 --img_size 128 --gpu_num 2 --progressive True --phase train


python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset rbc_500 --input_channels 2 --start_res 8 \
	--img_size 256 --gpu_num 4 --progressive True --phase train \
	--checkpoint_dir ../stored_outputs/rbc_working_with_std/checkpoint \
	--result_dir ../stored_outputs/rbc_working_with_std/result \
	--log_dir ../stored_outputs/rbc_working_with_std/log --sample_dir ../stored_outputs/rbc_working_with_std/sample  
 
