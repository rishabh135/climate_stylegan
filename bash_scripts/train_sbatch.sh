#!/bin/sh
#SBATCH -N 2
#SBATCH -c 20
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:4
#SBATCH -t 08:00:00 
	
#the minimum amount of time the job should run

module load esslurm
module load cuda/10.0.130
module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13
module load tensorflow/gpu-1.13.1-py36
pip install tensorflow-datasets



#run the application:
python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset FFHQ_128 --img_size 128 --gpu_num 4 --progressive True --phase train --checkpoint_dir ../stored_outputs/FFHQ_128_with_1200k_iteratrion_new/checkpoint --result_dir ../stored_outputs/FFHQ_128_with_1200k_iteratrion_new/result --log_dir ../stored_outputs/FFHQ_128_with_1200k_iteratrion_new/log --sample_dir ../stored_outputs/FFHQ_128_with_1200k_iteratrion_new/sample 	


# python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/main.py --dataset rbc_2000  --checkpoint_dir ../stored_outputs/rbc_2000/checkpoint --result_dir ../stored_outputs/rbc_2000/result --log_dir ../stored_outputs/rbc_2000/log --sample_dir ../stored_outputs/rbc_2000/sample   --img_size 128 --gpu_num 2 --progressive True --phase train