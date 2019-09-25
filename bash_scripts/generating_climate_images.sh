#!/bin/sh
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 00:5:00 

module load esslurm
module load cuda/10.0.130


# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/generating_climate_images.py --dataset climate_stylegan2900 --input_channels 1 --start_res 8 --test_num 50 --counter_number 159490 \
	--img_size 256 --gpu_num 8 --progressive True --phase test \
	--checkpoint_dir ./stored_outputs/wo_norm/checkpoint --result_dir ./stored_outputs/wo_norm/result \
	--log_dir ./stored_outputs/wo_norm/log --sample_dir ./stored_outputs/wo_norm/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/ \
	--name_experiment "generating climate images wo norm" 
 

