#!/bin/sh
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 00:09:00 

module load esslurm
module load cuda/10.0.130


# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 1 --start_res 512 \
	--img_size 512 --gpu_num 1 --progressive False --phase test  --fixed_offset -1 --batch_size 1 \
	--decay_logan True --feature_matching_loss True  --logan_flag True \
	--test_num 50 --inference_counter_number 230000 \
	--custom_cropping_flag False --decay_logan True --feature_matching_loss True  --logan_flag True  --wandb_flag True\
	--checkpoint_dir ./stored_outputs/training_omega_resolution_512_logan_feature_matching_without_normalization/checkpoint \
	--result_dir ./stored_outputs/training_omega_resolution_512_logan_feature_matching_without_normalization/result \
	--log_dir ./stored_outputs/training_omega_resolution_512_logan_feature_matching_without_normalization/log \
	--sample_dir ./stored_outputs/training_omega_resolution_512_logan_feature_matching_without_normalization/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/\
	--name_experiment "[Inference training_omega_resolution_512_logan_feature_matching_without_normalization]"
 







# --dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/


