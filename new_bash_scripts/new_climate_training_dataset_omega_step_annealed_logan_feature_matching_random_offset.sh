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
module load tensorflow/gpu-1.13.1-py36

# chekpoint_dir ./stored_outputs/wo_norm_512/checkpoint and counter 255468



# ./stored_outputs/fixed_offset_normalized_feature_matching_step_annealing_custom_cropping_logan_annealed_omega_128/checkpoint

#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 1 --start_res 8 \
	--img_size 128 --crop_size 128 --gpu_num 4 --progressive False --phase train  --fixed_offset 0\
	--custom_cropping_flag True --decay_logan True  --feature_matching_loss True  --logan_flag True\
	--checkpoint_dir ./stored_outputs/new_dataset_logan_annealed_omega_random_offset_128/checkpoint \
	--result_dir ./stored_outputs/new_dataset_logan_annealed_omega_random_offset_128/result \
	--log_dir ./stored_outputs/new_dataset_logan_annealed_omega_random_offset_128/log \
	--sample_dir ./stored_outputs/new_dataset_logan_annealed_omega_random_offset_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/ \
	--name_experiment "[Stylegan V1 with new dataset training with random offset logan optimization feature_matching ] normalized_climate data at 128 new_version"
 




