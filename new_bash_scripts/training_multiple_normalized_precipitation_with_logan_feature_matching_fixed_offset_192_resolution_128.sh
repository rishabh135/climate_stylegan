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
	--checkpoint_dir ./stored_outputs/precipitation_multiple_scale_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/checkpoint \
	--result_dir ./stored_outputs/precipitation_multiple_scale_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/result \
	--log_dir ./stored_outputs/precipitation_multiple_scale_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/log \
	--sample_dir ./stored_outputs/precipitation_multiple_scale_normalized_training_with_logan_feature_matching_fixed_offset_192_resolution_128_correct_chi/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_precipitation_3_normalized_channels/\
	--name_experiment "[Stylegan-V1-precipitation_normalized_precipitation_multiple_scale]"
 







# --dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_precipitation_normalized_0.008_x/
	



# ****** data from year for precipitation before normalization: 1996  [:,0:1, 128:640, 320:832] shape 
# (2920, 1, 512, 512), max 0.0257529579102993 min -1.0273492548913468e-16 std 0.00018855308007914573 m
# ean 4.3442880269140005e-05
#  data from year for precipitation after normalization: 1996  [:,0:3, 128:640, 320:832] shape (2920, 
# 3, 512, 512), max 0.9279356002807617 min -5.136745927702621e-14 std 0.029353804886341095 mean 0.0087
# 00607344508171


# Time taken to complete iteration 213.61285972595215


# ****** data from year for precipitation before normalization: 1997  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.026823148131370544 min -1.349253340060156e-16 std 0.00018518623255658895 mean 4.30329455411993e-05
#  data from year for precipitation after normalization: 1997  [:,0:3, 128:640, 320:832] shape (2920, 3, 512, 512), max 0.9306113123893738 min -6.74626641442716e-14 std 0.02857668697834015 mean 0.008683837950229645


# Time taken to complete iteration 225.1912305355072


# ****** data from year for precipitation before normalization: 1998  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.023110272362828255 min -1.1064042295220197e-16 std 0.00019681146659422666 mean 4.336315396358259e-05
#  data from year for precipitation after normalization: 1998  [:,0:3, 128:640, 320:832] shape (2920, 3, 512, 512), max 0.9203513264656067 min -5.5320209676155974e-14 std 0.029761452227830887 mean 0.008607693016529083


# Time taken to complete iteration 255.9586946964264


# ****** data from year for precipitation before normalization: 1999  [:,0:1, 128:640, 320:832] shape (2920, 1, 512, 512), max 0.025355152785778046 min -1.20826920585221e-16 std 0.0001803209015633911 mean 4.277197513147257e-05
#  data from year for precipitation after normalization: 1999  [:,0:3, 128:640, 320:832] shape (2920, 3, 512, 512), max 0.9268876314163208 min -6.041345944557755e-14 std 0.028932753950357437 mean 0.008612449280917645


# Time taken to complete iteration 243.55135536193848
