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
module load tensorflow/gpu-1.13.1-py36



python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/style_mixing_rbc.py \
	--dataset rbc_3500 --input_channels 2 --start_res 8 --test_num 400\
	--img_size 256 --gpu_num 1 --progressive True \
	--checkpoint_dir ../stored_outputs/one_seventh_divergence/checkpoint --result_dir ../stored_outputs/one_seventh_divergence/result \
	--log_dir ../stored_outputs/one_seventh_divergence/log --sample_dir ../stored_outputs/one_seventh_divergence/sample  \
	--name_experiment "style_mixing_rbc for one_seventh_divergence"
 
