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
	--custom_cropping_flag True --decay_logan False --feature_matching_loss True  --logan_flag False \
	--checkpoint_dir ./stored_outputs/only_ux_normalized_constant_10_4years_dataset_wo_logan_fixed_offset_192_resolution_128/checkpoint \
	--result_dir ./stored_outputs/only_ux_normalized_constant_10_4years_dataset_wo_logan_fixed_offset_192_resolution_128/result \
	--log_dir ./stored_outputs/only_ux_normalized_constant_10_4years_dataset_wo_logan_fixed_offset_192_resolution_128/log \
	--sample_dir ./stored_outputs/only_ux_normalized_constant_10_4years_dataset_wo_logan_fixed_offset_192_resolution_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_ux_normalized_by_constant_10/ \
	--name_experiment "[Only UX normalized_constant_10 _4years_dataset_wo_logan_fixed_offset_192 and resolution_128]"
 







# --dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_ux_normalized_by_constant_10/
# ****** data from year for only ux  before normalization: 1996  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 75.88934326171875 min -82.63091278076172 std 8.912748336791992 mean 1.4762574434280396
#  data from year for only ux after normalization: 1996  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.588934421539307 min -8.263091087341309 std 0.8912774324417114 mean 0.14762620627880096 


# Time taken to complete iteration 22.765774250030518


# ****** data from year for only ux  before normalization: 1997  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 70.5874252319336 min -80.48070526123047 std 8.641823768615723 mean 1.463395595550537
#  data from year for only ux after normalization: 1997  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.058742523193359 min -8.048070907592773 std 0.8641855120658875 mean 0.14634007215499878 


# Time taken to complete iteration 22.38157844543457


# ****** data from year for only ux  before normalization: 1998  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 72.6349105834961 min -74.80836486816406 std 9.014705657958984 mean 1.5494831800460815
#  data from year for only ux after normalization: 1998  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.263491153717041 min -7.480836391448975 std 0.9014723300933838 mean 0.154950350522995


# Time taken to complete iteration 22.279215335845947


# ****** data from year for only ux  before normalization: 1999  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 73.51978302001953 min -65.36072540283203 std 9.073304176330566 mean 1.2316417694091797
#  data from year for only ux after normalization: 1999  [:,4:5, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.351978302001953 min -6.536072731018066 std 0.9073291420936584 mean 0.12316352128982544 


# Time taken to complete iteration 22.461650371551514
# ~                                                                                  
