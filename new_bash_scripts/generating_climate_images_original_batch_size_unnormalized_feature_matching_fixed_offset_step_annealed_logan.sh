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
module load tensorflow/gpu-1.15.0-py37 

# chekpoint_dir ./stored_outputs/wo_norm_512/checkpoint and counter 255468

#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 7 --start_res 8 --test_num 50 --inference_counter_number 191718 \
# 	--img_size 512 --gpu_num 1 --progressive True --phase test \
# 	--checkpoint_dir ./stored_outputs/normalized_seven_channel_512/checkpoint --result_dir ./stored_outputs/normalized_seven_channel_512/result \
# 	--log_dir ./stored_outputs/normalized_seven_channel_512/log --sample_dir ./stored_outputs/normalized_seven_channel_512/sample \
# 	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized \
# 	--name_experiment "generating images for normalized_seven_channel_climate_data at 512 resolution"
 





python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 1 --test_num 50 --inference_counter_number 81250 \
	--img_size 128 --crop_size 128 --gpu_num 1 --progressive False --phase test  --fixed_offset 192\
	--custom_cropping_flag True --decay_logan True  --feature_matching_loss True  --logan_flag True\
	--checkpoint_dir ./stored_outputs/original_batch_size_normalized_fixed_offset_feature_matching_step_annealing_custom_cropping_logan_annealed_omega_128/checkpoint \
	--result_dir ./stored_outputs/original_batch_size_normalized_fixed_offset_feature_matching_step_annealing_custom_cropping_logan_annealed_omega_128/result \
	--log_dir ./stored_outputs/original_batch_size_normalized_fixed_offset_feature_matching_step_annealing_custom_cropping_logan_annealed_omega_128/log \
	--sample_dir ./stored_outputs/original_batch_size_normalized_fixed_offset_feature_matching_step_annealing_custom_cropping_logan_annealed_omega_128/sample \
	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized/ \
	--name_experiment "[SGAN inference counter 81250 normalized original batch size ]generating LoGAN for 1 channel"

