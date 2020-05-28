#!/bin/sh
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 00:3:00 

module load esslurm
module load cuda/10.0.130


# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36

# chekpoint_dir ./stored_outputs/wo_norm_512/checkpoint and counter 255468

#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 7 --start_res 8 --test_num 50 --inference_counter_number 191718 \
# 	--img_size 512 --gpu_num 1 --progressive True --phase test \
# 	--checkpoint_dir ./stored_outputs/normalized_seven_channel_512/checkpoint --result_dir ./stored_outputs/normalized_seven_channel_512/result \
# 	--log_dir ./stored_outputs/normalized_seven_channel_512/log --sample_dir ./stored_outputs/normalized_seven_channel_512/sample \
# 	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized \
# 	--name_experiment "generating images for normalized_seven_channel_climate_data at 512 resolution"
 


# 131718

#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 2 --start_res 8 --test_num 50 --inference_counter_number 216718 \
	--img_size 128 --gpu_num 1 --progressive True --phase test \
	--checkpoint_dir ./stored_outputs/2_channel_logan_wo_norm_128_crop_ux_uy/checkpoint --result_dir ./stored_outputs/2_channel_logan_wo_norm_128_crop_ux_uy/result \
	--log_dir ./stored_outputs/2_channel_logan_wo_norm_128_crop_ux_uy/log --sample_dir ./stored_outputs/2_channel_logan_wo_norm_128_crop_ux_uy/sample \
	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/unnormalized \
	--name_experiment " generating Ux and Uy together at 128 without normalization at generator 216718"
 


