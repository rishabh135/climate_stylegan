#!/bin/sh
#SBATCH -N 1
#SBATCH -c 80
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:8
#SBATCH -t 03:59:00 

module load esslurm
module load cuda/10.0.130



# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load tensorflow/gpu-1.15.0-py37 

#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 --input_channels 3 --start_res 8 \
	--img_size 512 --gpu_num 8 --progressive True --phase train \
	--checkpoint_dir ./stored_outputs/logan_three_channel_norm/checkpoint --result_dir ./stored_outputs/logan_three_channel_norm/result \
	--log_dir ./stored_outputs/logan_three_channel_norm/log --sample_dir ./stored_outputs/logan_seven_three_norm/sample \
	--dataset_location /project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized \
	--name_experiment "LoGAN normalized three channel climate data at 512"
 

