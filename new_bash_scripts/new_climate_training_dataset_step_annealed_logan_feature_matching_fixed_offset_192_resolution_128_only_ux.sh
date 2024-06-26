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
	--custom_cropping_flag True --decay_logan True  --feature_matching_loss True  --logan_flag True \
	--checkpoint_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/checkpoint \
	--result_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/result \
	--log_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/log \
	--sample_dir ./stored_outputs/new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_normalized_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original_normalized_ux_channel/ \
	--name_experiment "[Stylegan V1 with new_dataset_logan_annealed_feature_matching_fixed_offset_192_only_ux_128 dataset_location_climate_data_original_normalized_ux_channel ] normalized_climate data at 128 new_version"
 








# --dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original_normalized_ux_channel/ \


# ****** data from year for only ux  before normalization: 1996  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 75.88934326171875 min -82.63091278076172 std 8.912748336791992 mean 1.4762574434280396
#  data from year for only ux after normalization: 1996  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 8.349061965942383 min -9.436726570129395 std 1.0000011920928955 mean 9.460712817599415e-07 


# Time taken to complete iteration 24.767932415008545


# ****** data from year for only ux  before normalization: 1997  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 70.5874252319336 min -80.48070526123047 std 8.641823768615723 mean 1.463395595550537
#  data from year for only ux after normalization: 1997  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.998778343200684 min -9.482269287109375 std 1.0000056028366089 mean 9.964169294107705e-07 


# Time taken to complete iteration 25.050373792648315


# ****** data from year for only ux  before normalization: 1998  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 72.6349105834961 min -74.80836486816406 std 9.014705657958984 mean 1.5494831800460815
#  data from year for only ux after normalization: 1998  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.885496139526367 min -8.470365524291992 std 0.9999959468841553 mean 2.3472616703656968e-06


# Time taken to complete iteration 24.56243896484375


# ****** data from year for only ux  before normalization: 1999  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 73.51978302001953 min -65.36072540283203 std 9.073304176330566 mean 1.2316417694091797
#  data from year for only ux after normalization: 1999  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.967123985290527 min -7.339373588562012 std 0.9999983310699463 mean -3.107603276930604e-07


# Time taken to complete iteration 25.30447006225586


