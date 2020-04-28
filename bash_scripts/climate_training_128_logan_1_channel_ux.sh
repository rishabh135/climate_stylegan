#!/bin/sh
#SBATCH -N 1
#SBATCH -c 80
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:8
#SBATCH -t 03:59:00 

module load esslurm
module load cuda/10.0.130



# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 1 --start_res 8 \
	--img_size 128 --crop_size 128 --gpu_num 8 --progressive True --phase train \
	--checkpoint_dir ./stored_outputs/wo_norm_128_crop_one_channel_ux/checkpoint --result_dir ./stored_outputs/wo_norm_128_crop_one_channel_ux/result \
	--log_dir ./stored_outputs/wo_norm_128_crop_one_channel_ux/log --sample_dir ./stored_outputs/wo_norm_128_crop_one_channel_ux/sample \
	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/unnormalized \
	--name_experiment " Ux alone 128 cropped climate data without normalization on a single channel"
 



