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
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 1 --start_res 8 --batch_size 1 --fixed_offset 192 \
	--test_num 50 --inference_counter_number 94375 \
	--img_size 128 --crop_size 128 --gpu_num 1 --progressive False --phase test  --fixed_offset 192\
	--custom_cropping_flag True --decay_logan True  --feature_matching_loss True  --logan_flag True\
	--checkpoint_dir ./stored_outputs/new_dataset_logan_annealed_omega_128/checkpoint \
	--result_dir ./stored_outputs/new_dataset_logan_annealed_omega_128/result \
	--log_dir ./stored_outputs/new_dataset_logan_annealed_omega_128/log \
	--sample_dir ./stored_outputs/new_dataset_logan_annealed_omega_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/ \
	--name_experiment "[NEW Inference : fixed_offset_192 Feature matching step-annealed logan Omega 128 with custom cropping_normalized] wo_norm_climate data at 128 new_version"
 




