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




#run the application:
# python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
# 	--input_channels 1 --start_res 8 --batch_size 1 --fixed_offset 192 \
# 	--test_num 50 --inference_counter_number 60000 \
# 	--img_size 128 --crop_size 128 --gpu_num 1 --progressive False --phase test  --fixed_offset 192\
# 	--custom_cropping_flag True --decay_logan True  --feature_matching_loss True  --logan_flag True\
# 	--checkpoint_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/checkpoint \
# 	--result_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/result \
# 	--log_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/log \
# 	--sample_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/sample \
# 	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original_normalized_ux_channel/ \
# 	--name_experiment "[Inference : NEW Datset only ux_ Stylegan V1 with new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_128 dataset_location_climate_data_original_normalized_ux_channel ] normalized_climate data at 128 new_version"
 




#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 1 --start_res 8 --batch_size 1 \
	--test_num 50 --inference_counter_number 75000 \
	--img_size 128 --crop_size 128 --gpu_num 1 --progressive False --phase test  --fixed_offset 192\
	--custom_cropping_flag True --decay_logan True  --feature_matching_loss True  --logan_flag True\
	--checkpoint_dir ./stored_outputs/wo_normalized_ux_logan_annealed_feature_matching_fixed_offset_192_res_128/checkpoint \
	--result_dir ./stored_outputs/wo_normalized_ux_logan_annealed_feature_matching_fixed_offset_192_res_128/result \
	--log_dir ./stored_outputs/wo_normalized_ux_logan_annealed_feature_matching_fixed_offset_192_res_128/log \
	--sample_dir ./stored_outputs/wo_normalized_ux_logan_annealed_feature_matching_fixed_offset_192_res_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original_ux_channel/ \
	--name_experiment "[Inference Stylegan V1 only_ux_without_normalization  wo_normalized_ux_logan_annealed_feature_matching_fixed_offset_192_res_128 ] normalized_climate data at 128 new_version"
 




