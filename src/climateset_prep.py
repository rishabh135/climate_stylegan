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
















##########################################################################################
##########################################################################################


# Calculating normalized data with 4d shape across overall std 


##########################################################################################
##########################################################################################



def calculate_std_over_all_years():

	load_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h3.*.nc"

	files = sorted(glob(load_file_path))

	print("\n files found in the current directory for year  = {}".format( len(files)))
	list_of_omega_data = []    
	for _, file in enumerate(files):

		tmp_ds = xr.open_dataset(file, decode_times=False)
		list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))


	tt = time.time()
	numpy_array_of_omega = np.concatenate(list_of_omega_data, axis=0)
	mode = stats.mode(numpy_array_of_omega, axis=None)


	print("\n\n****** data from all years (1995-98)  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} mode {} and frequency of mode : {}  \n\n ".format( numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean(), mode[0], mode[1] ))

	return numpy_array_of_omega.std() 

##########################################################################################
##########################################################################################

#  statistics for the overall_std_normalized data 

##########################################################################################
##########################################################################################



# ****** data from all years (1995-98)  [:, 128:640, 320:832] shape (10568, 1, 512, 512), max 9.571378707885742 min -13.53
# 394603729248 std 0.20382769405841827 mean -0.0012632719008252025 mode [0.06919879] and frequency of mode : [124]



#  files found in the current directory for year 1995 = 334
# Time taken to complete iteration with shape (2672, 1, 1, 512, 512) in time 5.927042722702026

#  files found in the current directory for year 1996 = 365
# Time taken to complete iteration with shape (2920, 1, 1, 512, 512) in time 6.5855934619903564

#  files found in the current directory for year 1997 = 365
# Time taken to complete iteration with shape (2920, 1, 1, 512, 512) in time 6.580164909362793

#  files found in the current directory for year 1998 = 257
# Time taken to complete iteration with shape (2056, 1, 1, 512, 512) in time 4.311551570892334


##########################################################################################
##########################################################################################

#  statistics for the overall_std_normalized data 

##########################################################################################
##########################################################################################


def climate_data_with_std_norm_4d(save_dir_path, list_of_years):


	""" saving omega data as original without any normalization"""
	t=time.time()

	overall_std = calculate_std_over_all_years()


	for year in list_of_years:
		
		load_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/CAM5-1-0.25degree_All-Hist_est1_v3_run1.cam.h3.{}-*.nc".format(year)
		
		files = sorted(glob(load_file_path))

		print("\n files found in the current directory for year {} = {}".format(year, len(files)))
		list_of_omega_data = []    
		for _, file in enumerate(files):

			tmp_ds = xr.open_dataset(file, decode_times=False)
			list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))

		tt = time.time()

		numpy_array_of_omega = np.true_divide(np.concatenate(list_of_omega_data, axis=0), overall_std)


		# mode = stats.mode(numpy_array_of_omega, axis=None)
		
		
		# print("\n\n****** data from year {}  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} mode {} and frequency of mode : {}  \n\n ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean(), mode[0], mode[1] ))
		
		save_path = os.path.join(save_dir_path , "climate_data_normalized_overall_std/")

		if not os.path.exists(save_path):
			os.makedirs(save_path)

		numpy_array_of_omega = np.expand_dims(numpy_array_of_omega, axis= 1)

		np.save( str(save_path) + "{}.npy".format(year), numpy_array_of_omega)
		print("Time taken to complete iteration with shape {} in time {} ".format( numpy_array_of_omega.shape ,time.time() - tt))
		



























##########################################################################################
##########################################################################################


# Overall Driving functions with relevant years and the save path for the datasets 


##########################################################################################
##########################################################################################












def climate_data_wo_norm_4d_npy(save_dir_path, list_of_years):


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















































		***** data from year 1996  [:, 128:640, 320:832] shape (2920, 512, 512), max 9.571378707885742 min -12.924219131469727 std 0.20633453130722046 mean -0.001688328804448247


		ime taken to complete iteration 35.538962841033936


		***** data from year 1997  [:, 128:640, 320:832] shape (2920, 512, 512), max 8.109827995300293 min -13.53394603729248 std 0.20219865441322327 mean -0.001156140468083322


		ime taken to complete iteration 35.89456605911255


		***** data from year 1998  [:, 128:640, 320:832] shape (2920, 512, 512), max 8.023037910461426 min -12.149008750915527 std 0.20486298203468323 mean -0.0006242241361178458


		ime taken to complete iteration 35.77512311935425


		***** data from year 1999  [:, 128:640, 320:832] shape (2920, 512, 512), max 7.709518909454346 min -11.99194622039795 std 0.19934770464897156 mean -0.0010409714886918664


		ime taken to complete iteration 36.906309604644775


		***** data from year 2000  [:, 128:640, 320:832] shape (2920, 512, 512), max 8.345470428466797 min -15.302850723266602 std 0.2007954716682434 mean -0.00103561335708946


		slurm-814200.out" 32L, 1256C                                                                                                                                     1,1           Top








	"""



	""" saving omega data as original without any normalization"""
	t=time.time()
	for year in list_of_years:
		
		load_file_path = "/project/projectdirs/dasrepo/mustafa/data/climate/sims/normalized/normalized_seven_channels_{}.npy".format(year)
		
		# files = sorted(glob(load_file_path))

		# print("\n files found in the current directory for year {} = {}".format(year, len(files)))
		# list_of_omega_data = []    
		# for _, file in enumerate(files):

		# 	tmp_ds = xr.open_dataset(file, decode_times=False)
		# 	list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))

		tt = time.time()
	


		numpy_array_of_omega = np.load(load_file_path)[: , -1:, :, :]
		

		print("\n\n****** data from year {}  [:,1, 128:640, 320:832] shape {}, max {} min {} std {} mean {} \n\n ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean() ))
		
		save_path = os.path.join(save_dir_path , "climate_data_original/")

		if not os.path.exists(save_path):
			os.makedirs(save_path)

		np.save( str(save_path) + "{}.npy".format(year), numpy_array_of_omega)
		print("Time taken to complete iteration", time.time() - tt)
		
	
	return 



save_dir_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/"
list_of_years = ["1996", "1997", "1998"]
print("Started running the program")


# climate_data_with_std_norm_4d(save_dir_path, list_of_years)

# climate_data_wo_norm_4d(save_dir_path, list_of_years)


climate_data_wo_norm_4d_npy(save_dir_path, list_of_years)








"""




load_file_path = "/project/projectdirs/dasrepo/mustafa/data/climate/sims/unnormalized/seven_channels_{}.npy".format(year)

****** data from year 1996  [:, 1, 128:640, 320:832] shape (2920, 7, 512, 512), max 105821.8125 min -82.63091278076172 std 35349.2109375 mean 14508.9345703125


Time taken to complete iteration 50.24332547187805


****** data from year 1997  [:, 1, 128:640, 320:832] shape (2920, 7, 512, 512), max 106334.515625 min -80.48070526123047 std 35358.796875 mean 14512.7763671875


Time taken to complete iteration 53.00413680076599


****** data from year 1998  [:, 1, 128:640, 320:832] shape (2920, 7, 512, 512), max 106354.359375 min -77.23316955566406 std 35359.5703125 mean 14514.5615234375


Time taken to complete iteration 56.13152599334717


****** data from year 1999  [:, 1, 128:640, 320:832] shape (2920, 7, 512, 512), max 106225.515625 min -66.47136688232422 std 35366.79296875 mean 14517.2666015625


Time taken to complete iteration 54.4095184803009


****** data from year 2000  [:, 1, 128:640, 320:832] shape (2920, 7, 512, 512), max 106174.21875 min -76.54035949707031 std 35365.5 mean 14517.6044921875


Time taken to complete iteration 55.64541745185852




"""