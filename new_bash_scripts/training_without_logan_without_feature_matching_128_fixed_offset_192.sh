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
	--input_channels 1 --start_res 8 \
	--img_size 128 --crop_size 128 --gpu_num 4 --progressive False --phase train  --fixed_offset 192 \
	--custom_cropping_flag True --decay_logan False  --feature_matching_loss False  --logan_flag False \
	--checkpoint_dir ./stored_outputs/training_without_logan_without_feature_matching_128_fixed_offset_192/checkpoint \
	--result_dir ./stored_outputs/training_without_logan_without_feature_matching_128_fixed_offset_192/result \
	--log_dir ./stored_outputs/training_without_logan_without_feature_matching_128_fixed_offset_192/log \
	--sample_dir ./stored_outputs/training_without_logan_without_feature_matching_128_fixed_offset_192/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/ \
	--name_experiment "[Stylegan V1 training_without_logan_without_feature_matching_128_fixed_offset_192] normalized_climate data at 128 new_version"
 








# --dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original_ux_channel/



# ****** data from year for only ux  before normalization: 1996  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 75.88934326171875 min -82.63091278076172 std 8.912748336791992 mean 1.4762574434280396
#  data from year for only ux after normalization: 1996  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 75.88934326171875 min -82.63091278076172 std 8.912748336791992 mean 1.4762574434280396


# Time taken to complete iteration 38.772714376449585


# ****** data from year for only ux  before normalization: 1997  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 70.5874252319336 min -80.48070526123047 std 8.641823768615723 mean 1.463395595550537
#  data from year for only ux after normalization: 1997  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 70.5874252319336 min -80.48070526123047 std 8.641823768615723 mean 1.463395595550537


# Time taken to complete iteration 40.0781033039093


# ****** data from year for only ux  before normalization: 1998  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 72.6349105834961 min -74.80836486816406 std 9.014705657958984 mean 1.5494831800460815
#  data from year for only ux after normalization: 1998  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 72.6349105834961 min -74.80836486816406 std 9.014705657958984 mean 1.5494831800460815


# Time taken to complete iteration 39.133992433547974


# ****** data from year for only ux  before normalization: 1999  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 73.51978302001953 min -65.36072540283203 std 9.073304176330566 mean 1.2316417694091797
#  data from year for only ux after normalization: 1999  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 73.51978302001953 min -65.36072540283203 std 9.073304176330566 mean 1.2316417694091797


# Time taken to complete iteration 39.70984768867493




