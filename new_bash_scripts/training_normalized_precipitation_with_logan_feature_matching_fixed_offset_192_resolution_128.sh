#!/bin/sh
#SBATCH -N 1
#SBATCH -c 10
#SBATCH -C gpu
#SBATCH -A dasrepo
#SBATCH --gres=gpu:1
#SBATCH -t 01:59:00 

module load esslurm
module load cuda/10.0.130


# module load cudatoolkit/10.0.130_3.22-7.0.0.1_3.3__gdfb4ce5
# module load nccl
module load python/3.7-anaconda-2019.07
source activate tf-1.13


#run the application:
python  /global/cscratch1/sd/rgupta2/backup/climate_stylegan/src/main.py --dataset climate_stylegan2900 \
	--input_channels 1 --start_res 8 \
	--img_size 128 --crop_size 128 --gpu_num 1 --progressive False --phase train  --fixed_offset 192 \
	--custom_cropping_flag True --decay_logan True --feature_matching_loss True  --logan_flag True \
	--checkpoint_dir ./stored_outputs/precipitation_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128/checkpoint \
	--result_dir ./stored_outputs/precipitation_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128/result \
	--log_dir ./stored_outputs/precipitation_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128/log \
	--sample_dir ./stored_outputs/precipitation_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_precipitation_normalized_0.008_x/\
	--name_experiment "[Stylegan-V1-precipitation_normalized_with_008]"
 







# --dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_precipitation_normalized_0.008_x/
	

# Started running the program


# ****** data from year for precipitation before normalization: 1996  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.0257529579102993 min -1.0273492548913468e-16 std 0.00018855308007914573
#  mean 4.3442880269140005e-05
#  data from year for precipitation after normalization: 1996  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.7629836797714233 min -1.2841864819256553e-14 std 0.016680249944329262 mean 0.0
# 050241523422300816


# Time taken to complete iteration 23.737550497055054


# ****** data from year for precipitation before normalization: 1997  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.026823148131370544 min -1.349253340060156e-16 std 0.0001851862325565889
# 5 mean 4.30329455411993e-05
#  data from year for precipitation after normalization: 1997  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.7702677249908447 min -1.68656660360679e-14 std 0.016169380396604538 mean 0.004
# 99301590025425


# Time taken to complete iteration 23.636591911315918


# ****** data from year for precipitation before normalization: 1998  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.023110272362828255 min -1.1064042295220197e-16 std 0.000196811466594226
# 66 mean 4.336315396358259e-05
#  data from year for precipitation after normalization: 1998  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.7428502440452576 min -1.3830052419038993e-14 std 0.017121024429798126 mean 0.0
# 04988004919141531


# Time taken to complete iteration 23.600712060928345


# ****** data from year for precipitation before normalization: 1999  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.025355152785778046 min -1.20826920585221e-16 std 0.0001803209015633911 
# mean 4.277197513147257e-05
#  data from year for precipitation after normalization: 1999  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.7601569890975952 min -1.5103364861394387e-14 std 0.01630186103284359 mean 0.00
# 496435072273016


# Time taken to complete iteration 23.756458044052124
# ~                                                                                                                                                                                                     
