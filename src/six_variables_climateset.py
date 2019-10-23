#!/bin/env python


import sys,os,os.path,time
# sys.path.append(os.path.expanduser('/global/u1/r/rgupta2/.local/lib/python3.7/site-packages/'))

# export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/site-packages"
import numpy as np
import time
from glob import glob
from tqdm import tqdm, trange
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr

from scipy import stats

t = time.time()




def climate_data_all_six_vars_wo_norm_4d(save_dir_path, year):

	""" saving omega data as original without any normalization"""
	t=time.time()

	six_vars = [ "pr","prw", "psl", "ts", "ua850", "va850"]
	suffix_different = ["pr", "ua850", "va850"]

	tarfile_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/all_6_variables_nc_files/"

	for channel in six_vars:
		tt = time.time()
		if(channel == "pr"):
			suffix = "12312359"
		else:
			suffix = "12312100"
		file = tarfile_path + "{}/{}_A3hr_CAM5-1-2-025degree_All-Hist_est1_v1-0_run001_{}01010000-{}{}.nc".format(channel, channel, year, year, suffix)    
		ds = xr.open_dataset(file, decode_times=False)
		if(channel in suffix_different):
			data = ds["{}".format(channel)][:, 0, 128:640, 320:832].values
		else:
			data = ds["{}".format(channel)][:, 128:640, 320:832].values
		print(data.shape)
		ds.close()
		save_path = os.path.join(save_dir_path , "climate_data_original_six_vars/{}/".format(channel))
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		np.save( str(save_path) + "{}_{}.npy".format(channel, year), data)
		print("Time taken to complete iteration", time.time() - tt)
		print("\n####################################\n")
































##########################################################################################
##########################################################################################


# Overall Driving functions with relevant years and the save path for the datasets 


##########################################################################################
##########################################################################################



	
list_of_years = [ "1998", "1999", "2000"]

print("Started running the program")
for year in list_of_years:
	climate_data_all_six_vars_wo_norm_4d(save_dir_path, year)

# climate_data_with_std_norm_4d(save_dir_path, list_of_years)

# climate_data_wo_norm_4d(save_dir_path, list_of_years)




