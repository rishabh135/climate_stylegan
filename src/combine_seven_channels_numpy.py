from __future__ import absolute_import, division, print_function, unicode_literals

import sys,os,os.path,time



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (15,20)



from tqdm import tqdm, trange
import re
import numpy as np
import PIL.Image

import xarray as xr
from glob import glob



six_vars = ["pr", "prw", "psl", "ts", "ua850", "va850", "omega"]
# tarfile_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/all_6_variables_nc_files/"

# years = np.arange(1996, 2016)

# file_names = tarfile_path + "{}_A3hr_CAM5-1-2-025degree_All-Hist_est1_v1-0_run001_{}01010000-{}12312100.nc".format(channel,year,year)
# [:, 128:640, 320:832, :]


list_of_years = [ "1998", "1999", "2000"]

for year in list_of_years:

	list_of_channels = []
	tt = time.time()
	for channel in tqdm(six_vars):

		save_dir_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/"
		save_path = os.path.join(save_dir_path , "climate_data_original_six_vars/{}/".format(channel))
		load_path = str(save_path) + "{}_{}.npy".format(channel, year)

		one_channel_data =  np.load(load_path, mmap_mode='r+')
		if( channel == "omega"):
			one_channel_data = np.squeeze(one_channel_data)
		
		print("one channel shape {} ".format(one_channel_data.shape)) 
		list_of_channels.append(one_channel_data)

	combined_channels = np.stack(list_of_channels, axis=1)

	save_path = os.path.join(save_dir_path , "climate_data_seven_channels_by_year/")

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	np.save( str(save_path) + "seven_channels_{}.npy".format(year), combined_channels)
	print("Time taken to complete iteration with shape {} in time {} ".format( combined_channels.shape ,time.time() - tt))
	
	print("\n####################################\n")

