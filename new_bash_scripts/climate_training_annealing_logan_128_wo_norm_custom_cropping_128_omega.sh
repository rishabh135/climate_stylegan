#!/bin/sh
#SBATCH -N 1
#SBATCH -c 20
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:2
#SBATCH -t 04:00:00 

module load esslurm
module load cuda/10.0.130



# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 1 --start_res 8 \
	--img_size 128 --crop_size 128 --gpu_num 2 --progressive False --phase train \
	--custom_cropping_flag True --decay_logan True \
	--checkpoint_dir ./stored_outputs/wo_norm_omega_128_2/checkpoint \
	--result_dir ./stored_outputs/wo_norm_omega_128_2/result \
	--log_dir ./stored_outputs/wo_norm_omega_128_2/log \
	--sample_dir ./stored_outputs/wo_norm_omega_128_2/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/ \
	--name_experiment "[SGAN1  Annealed logan Omega 128 with custom cropping] wo_norm_climate data at 128 new_version"
 



