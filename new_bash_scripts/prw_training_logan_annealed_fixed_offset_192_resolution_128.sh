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
	--img_size 128 --crop_size 128 --gpu_num 4 --progressive False --phase train  --fixed_offset 192\
	--custom_cropping_flag True --decay_logan True --feature_matching_loss True  --logan_flag True \
	--checkpoint_dir ./stored_outputs/prw_with_logan_feature_matching_fixed_offset_192_resolution_128/checkpoint \
	--result_dir ./stored_outputs/prw_with_logan_feature_matching_fixed_offset_192_resolution_128/result \
	--log_dir ./stored_outputs/prw_with_logan_feature_matching_fixed_offset_192_resolution_128/log \
	--sample_dir ./stored_outputs/prw_with_logan_feature_matching_fixed_offset_192_resolution_128/sample \
	--dataset_location /global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/only_prw/\
	--name_experiment "[prw_channel]"
 



# ****** data from year for prw before normalization: 1996  [:,1:2, 128:640, 320:832] shape (2920, 1, 512, 512), max 98.35570526123047 min 0.1284041404724121 std 15.178851127624512 mean 26.101123809814453
# Time taken to complete iteration 83.46162295341492


# ****** data from year for prw before normalization: 1997  [:,1:2, 128:640, 320:832] shape (2920, 1, 512, 512), max 95.14827728271484 min 0.12289127707481384 std 15.323355674743652 mean 26.503129959106445
# Time taken to complete iteration 87.63908672332764


# ****** data from year for prw before normalization: 1998  [:,1:2, 128:640, 320:832] shape (2920, 1, 512, 512), max 95.66651916503906 min 0.15732792019844055 std 15.778414726257324 mean 26.699684143066406
# Time taken to complete iteration 86.42332744598389


# ****** data from year for prw before normalization: 1999  [:,1:2, 128:640, 320:832] shape (2920, 1, 512, 512), max 97.20494842529297 min 0.17028875648975372 std 15.024084091186523 mean 26.208860397338867
# Time taken to complete iteration 91.41351175308228
# ~                                                                                                   
# ~                                                                                    

