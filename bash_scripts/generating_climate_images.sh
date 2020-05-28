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
 




#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 7 --start_res 8 --test_num 50 --inference_counter_number 266718 \
# 	--img_size 512 --gpu_num 1 --progressive True --phase test \
# 	--checkpoint_dir ./stored_outputs/logan_seven_channel_norm/checkpoint --result_dir ./stored_outputs/logan_seven_channel_norm/result \
# 	--log_dir ./stored_outputs/logan_seven_channel_norm/log --sample_dir ./stored_outputs/logan_seven_channel_norm/sample \
# 	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized \
# 	--name_experiment "generating LoGAN normalized generated images seven channel climate data at 512 for 266718"
 


# python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 1 --start_res 8 --test_num 50 --inference_counter_number 266718 \
# 	--img_size 512 --gpu_num 1 --progressive True --phase test \
# 	--checkpoint_dir ./stored_outputs/logan_seven_channel_norm/checkpoint --result_dir ./stored_outputs/logan_seven_channel_norm/result \
# 	--log_dir ./stored_outputs/logan_seven_channel_norm/log --sample_dir ./stored_outputs/logan_seven_channel_norm/sample \
# 	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized \
# 	--name_experiment "generating LoGAN normalized generated images seven channel climate data at 512 for 266718"





# python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 3 --start_res 8 --test_num 50 --inference_counter_number 267968 \
# 	--img_size 512 --gpu_num 1 --progressive True --phase test \
# 	--checkpoint_dir ./stored_outputs/logan_three_channel_norm/checkpoint --result_dir ./stored_outputs/logan_three_channel_norm/result \
# 	--log_dir ./stored_outputs/logan_three_channel_norm/log --sample_dir ./stored_outputs/logan_three_channel_norm/sample \
# 	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized \
# 	--name_experiment "generating LoGAN normalized for 3 channel 267968"


python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 1 --start_res 8 --test_num 50 --inference_counter_number 246718 \
	--img_size 512 --gpu_num 1 --progressive True --phase test \
	--checkpoint_dir ./stored_outputs/wo_norm_512_logan/checkpoint --result_dir ./stored_outputs/wo_norm_512_logan/result \
	--log_dir ./stored_outputs/wo_norm_512_logan/log --sample_dir ./stored_outputs/wo_norm_512_logan/sample \
	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/unnormalized \
	--name_experiment "generating LoGAN without normalized for 1 channel from wo_norm_512_logan 246718"
