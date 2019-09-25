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





def climate_data_wo_norm(save_dir_path, list_of_years): 


	""" saving omega data as original without any normalization"""

	for year in list_of_years:
		
		load_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h3.{}-*.nc".format(year)
		
		files = sorted(glob(load_file_path))
		list_of_omega_data = []
		print("Time takes : ", time.time()-t)    
		for _, file in tqdm(enumerate(files)):
			t=time.time()
			tmp_ds = xr.open_dataset(file, decode_times=False)
			list_of_omega_data.append(tmp_ds["OMEGA500"][:, 128:640, 320:832].values)

		tt = time.time()
		numpy_array_of_omega = np.concatenate(list_of_omega_data, axis=0)
		
		print(" data from year {}  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean() ))
		
		save_path = os.path.join(save_dir_path , "climate_data_original/")

		if not os.path.exists(save_path):
			os.makedirs(save_path)

		np.save( str(save_path) + "{}.npy".format(year), numpy_array_of_omega)
		print("Time taken to complete iteration", time.time() - tt)
		

		
		
	"""
	Time takes :  0.06397104263305664

	data from year 1995  [:, 128:640, 320:832] shape (2672, 512, 512), max 6.6639084815979 min -12.746827125549316 std 0.20175360143184662 mean -0.0011749943951144814

	Time taken to complete iteration 4.458622694015503






	Time takes :  5.034161329269409

	data from year 1996  [:, 128:640, 320:832] shape (2920, 512, 512), max 9.571378707885742 min -12.924219131469727 std 0.20633453130722046 mean -0.001688328804448247

	Time taken to complete iteration 5.58919882774353






	Time takes :  6.016656160354614

	data from year 1997  [:, 128:640, 320:832] shape (2920, 512, 512), max 8.109827995300293 min -13.53394603729248 std 0.20219865441322327 mean -0.001156140468083322

	Time taken to complete iteration 6.169508934020996






	Time takes :  6.677528142929077

	data from year 1998  [:, 128:640, 320:832] shape (2056, 512, 512), max 7.37013053894043 min -11.905616760253906 std 0.20523402094841003 mean -0.0009264641557820141

	Time taken to complete iteration 3.767068862915039
															 

	"""





def climate_data_wo_norm_4d(save_dir_path, list_of_years):


	"""



	 files found in the current directory for year 1995 = 334


	****** data from year 1995  [:, 128:640, 320:832] shape (2672, 1, 512, 512), max 6.6639084815979 min -12.746827125549316 std 0.20175360143184662 mean -0.0011749943951144814 

	 
	Time taken to complete iteration 4.662921905517578

	 files found in the current directory for year 1996 = 365


	****** data from year 1996  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 9.571378707885742 min -12.924219131469727 std 0.20633453130722046 mean -0.001688328804448247 

	 
	Time taken to complete iteration 7.624687910079956

	 files found in the current directory for year 1997 = 365


	****** data from year 1997  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 8.109827995300293 min -13.53394603729248 std 0.20219865441322327 mean -0.001156140468083322 

	 
	Time taken to complete iteration 5.735872507095337

	 files found in the current directory for year 1998 = 257


	****** data from year 1998  [:, 128:640, 320:832] shape (2056, 1, 512, 512), max 7.37013053894043 min -11.905616760253906 std 0.20523402094841003 mean -0.0009264641557820141 

	 
	Time taken to complete iteration 5.970208168029785








	"""



	""" saving omega data as original without any normalization"""
	t=time.time()
	for year in list_of_years:
		
		load_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h3.{}-*.nc".format(year)
		
		files = sorted(glob(load_file_path))

		print("\n files found in the current directory for year {} = {}".format(year, len(files)))
		list_of_omega_data = []    
		for _, file in enumerate(files):

			tmp_ds = xr.open_dataset(file, decode_times=False)
			list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))

		tt = time.time()
		numpy_array_of_omega = np.concatenate(list_of_omega_data, axis=0)
		
		print("\n\n****** data from year {}  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} \n\n ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean() ))
		
		save_path = os.path.join(save_dir_path , "climate_data_original/")

		if not os.path.exists(save_path):
			os.makedirs(save_path)

		np.save( str(save_path) + "{}.npy".format(year), numpy_array_of_omega)
		print("Time taken to complete iteration", time.time() - tt)
		

		
		
	"""
	Time takes :  0.06397104263305664

	data from year 1995  [:, 128:640, 320:832] shape (2672, 512, 512), max 6.6639084815979 min -12.746827125549316 std 0.20175360143184662 mean -0.0011749943951144814

	Time taken to complete iteration 4.458622694015503






	Time takes :  5.034161329269409

	data from year 1996  [:, 128:640, 320:832] shape (2920, 512, 512), max 9.571378707885742 min -12.924219131469727 std 0.20633453130722046 mean -0.001688328804448247

	Time taken to complete iteration 5.58919882774353






	Time takes :  6.016656160354614

	data from year 1997  [:, 128:640, 320:832] shape (2920, 512, 512), max 8.109827995300293 min -13.53394603729248 std 0.20219865441322327 mean -0.001156140468083322

	Time taken to complete iteration 6.169508934020996






	Time takes :  6.677528142929077

	data from year 1998  [:, 128:640, 320:832] shape (2056, 512, 512), max 7.37013053894043 min -11.905616760253906 std 0.20523402094841003 mean -0.0009264641557820141

	Time taken to complete iteration 3.767068862915039
															 

	"""











def climate_data_with_max_norm_4d(save_dir_path, list_of_years):


	""" saving omega data as original without any normalization"""
	t=time.time()
	for year in list_of_years:
		
		load_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h3.{}-*.nc".format(year)
		
		files = sorted(glob(load_file_path))

		print("\n files found in the current directory for year {} = {}".format(year, len(files)))
		list_of_omega_data = []    
		for _, file in enumerate(files):

			tmp_ds = xr.open_dataset(file, decode_times=False)
			list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))

		tt = time.time()
		numpy_array_of_omega = np.concatenate(list_of_omega_data, axis=0)

		mode = stats.mode(numpy_array_of_omega, axis=None)
		
		
		print("\n\n****** data from year {}  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} mode {} and frequency of mode : {}  \n\n ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean(), mode[0], mode[1] ))
		
		# save_path = os.path.join(save_dir_path , "climate_data_original/")

		# if not os.path.exists(save_path):
		# 	os.makedirs(save_path)

		# np.save( str(save_path) + "{}.npy".format(year), numpy_array_of_omega)
		# print("Time taken to complete iteration", time.time() - tt)
		


save_dir_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/"
list_of_years = ["1995", "1996", "1997", "1998"]
print("Started running the program")

climate_data_with_max_norm_4d(save_dir_path, list_of_years)

# climate_data_wo_norm_4d(save_dir_path, list_of_years)



