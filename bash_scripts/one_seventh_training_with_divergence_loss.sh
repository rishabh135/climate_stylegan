#!/bin/sh
#SBATCH -N 1
#SBATCH -c 80
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:8
#SBATCH -t 08:00:00 

module load esslurm
module load cuda/10.0.130



# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07

source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36

pip install comet-ml



#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset FFHQ_128 --img_size 128 --gpu_num 2 --progressive True --phase train
python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset rbc_3500 --input_channels 2 \
	--start_res 8 --divergence_loss True \
	--dataset_location "/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/one_seventh/rbc_3500/noramlized_by_max/" \
	--img_size 256 --gpu_num 8 --progressive True --phase train \
	--checkpoint_dir ../stored_outputs/one_seventh_divergence/checkpoint --result_dir ../stored_outputs/one_seventh_divergence/result \
	--log_dir ../stored_outputs/one_seventh_divergence/log --sample_dir ../stored_outputs/one_seventh_divergence/sample \
	--name_experiment "one_seventh_max_normalized starts at 3500 and added divergence loss"
 
