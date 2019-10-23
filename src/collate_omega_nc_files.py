#!/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import sys,os,os.path,time
sys.path.append(os.path.expanduser('/global/u1/r/rgupta2/.local/lib/python3.7/site-packages/'))

# export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/site-packages"
import numpy as np
import time


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# %matplotlib inline
# plt.rcParams['figure.figsize'] = (15,20)

from tqdm import tqdm, trange
import time, re, sys, os
import numpy as np
import PIL.Image

import xarray as xr
from glob import glob


list_of_years = ["1998", "1999", "2000"]  



# Collating new CAM5 files give by karthik_ from directory /global/cscratch1/sd/karthik_/CAM5.1_0.25degree/1996to2000/

t=time.time()
for year in list_of_years:
	load_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/1996to2000/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h3.{}-*-00000.nc".format(year)
	files = sorted(glob(load_file_path))
	print("\n files found in the current directory for year {} = {}".format(year, len(files)))
	list_of_omega_data = []    
	for _, file in tqdm(enumerate(files)):
		tmp_ds = xr.open_dataset(file, decode_times=False)
		list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))
		tmp_ds.close()
	tt = time.time()
	numpy_array_of_omega = np.concatenate(list_of_omega_data, axis=0)
	print("\n\n****** data from year {}  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} \n\n ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean() ))
	
	channel = "omega"
	save_dir_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/"
	save_path = os.path.join(save_dir_path , "climate_data_original_six_vars/{}/".format(channel))
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	np.save( str(save_path) + "{}_{}.npy".format(channel, year), numpy_array_of_omega)
	print("Time taken to complete iteration", time.time() - tt)





#  files found in the current directory for year 1998 = 365
# 365it [04:06,  1.49it/s]


# ****** data from year 1998  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 8.023037910461426 min -12.1490087509155
# 27 std 0.20486298203468323 mean -0.0006242241361178458 

 
# Time taken to complete iteration 11.340159177780151

#  files found in the current directory for year 1999 = 365
# 365it [04:07,  1.61it/s]


# ****** data from year 1999  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.709518909454346 min -11.9919462203979
# 5 std 0.19934770464897156 mean -0.0010409714886918664 

 
# Time taken to complete iteration 7.848862648010254

#  files found in the current directory for year 2000 = 365
# 365it [04:11,  1.50it/s]


# ****** data from year 2000  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 8.345470428466797 min -15.3028507232666
# 02 std 0.2007954716682434 mean -0.00103561335708946 

 
# Time taken to complete iteration 8.634740591049194
# 