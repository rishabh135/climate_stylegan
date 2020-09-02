#!/bin/sh
#SBATCH -N 1
#SBATCH -c 40
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:4
#SBATCH -t 03:59:00 

module load esslurm
module load cuda/10.0.130


# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 3 --start_res 8 \
	--img_size 128 --crop_size 128 --gpu_num 4 --progressive False --phase train  --fixed_offset 192 \
	--custom_cropping_flag True --decay_logan True --feature_matching_loss True  --logan_flag True \
	--checkpoint_dir ./stored_outputs/precipitation_multiple_scale_1_unnormalized_2_normalized_channel_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/checkpoint \
	--result_dir ./stored_outputs/precipitation_multiple_scale_1_unnormalized_2_normalized_channel_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/result \
	--log_dir ./stored_outputs/precipitation_multiple_scale_1_unnormalized_2_normalized_channel_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/log \
	--sample_dir ./stored_outputs/precipitation_multiple_scale_1_unnormalized_2_normalized_channel_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_precipitation_1_unnormalized_2_normalized_channels/\
	--name_experiment "[pr_multiple_scale_1_unnormalized_2_normalized_channel]"
 





