#!/bin/env python


import sys,os,os.path,time

import numpy as np
import time


import xarray as xr

import tensorflow as tf
# tf.enable_eager_execution()


# from mpl_toolkits.basemap import Basemap


file_path = "/project/projectdirs/dasrepo/mustafa/datasets/Rayleigh_Benard/result_rb_2d__Ra_2.5e8__Pr_0.71__maxMach_0.1__t_D_max_diffusive_scaling__0.4.nc"


t = time.time()

print("Started reading the big xr file")
ds = xr.open_dataset(file_path, decode_times=False	)
print("Time takes : ", time.time()-t)








"""

To collect maximum normalized rbc data

"""


save_dir_path = "/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/one_seventh/"

number_of_timesteps = 3500
st_time = 3500
end_time = 23630
image_size = 256

ux_data_whole_max = ds['u_x'][st_time : end_time].values.max()
uy_data_whole_max = ds['u_y'][st_time : end_time].values.max()
max_value = max(ux_data_whole_max, uy_data_whole_max)


tt = time.time()







for start_time in range(st_time, end_time, number_of_timesteps):


	ux_data = ds['u_x'][start_time : start_time+ number_of_timesteps].values
	uy_data = ds['u_y'][start_time : start_time+ number_of_timesteps].values



	print(ux_data.shape)


	ux_std = ux_data.std()
	uy_std = uy_data.std()

	u_data_max= max( ux_data.max(), uy_data.max())

	print("Ux_data_max = {} and Uy data max = {} and u_data_max {}".format(ux_data.max(), uy_data.max(), u_data_max))
	print("Ux std = {} and Uy std = {}".format(ux_std, uy_std))

	
	print("Time taken to read all", time.time() - tt)


	# strides =  [x * ux_data.itemsize for x in [ux_data.shape[-1] * ux_data.shape[-2], image_size, ux_data.shape[-1], 1]]
	# print("strides : {} ".format(strides))
	# u_x = np.lib.stride_tricks.as_strided(ux_data, (number_of_timesteps, ux_data.shape[-1]//image_size, image_size, image_size), strides)




	# strides =  [x * uy_data.itemsize for x in [uy_data.shape[-1] * uy_data.shape[-2], image_size, uy_data.shape[-1], 1]]
	# print("strides : {} ".format(strides))
	# u_y = np.lib.stride_tricks.as_strided(uy_data, (number_of_timesteps, ux_data.shape[-1]//image_size, image_size, image_size), strides)


	# u_x = u_x.reshape( -1, image_size,image_size)
	# u_y = u_y.reshape( -1, image_size,image_size)

	# print("u-x shape {}  u_y shape  {} ".format(u_x.shape, u_y.shape))


	u_x = ux_data[:,:, :256]
	u_y = uy_data[:,:, :256]



	channels = np.array([u_x, u_y])
	print(channels.shape)
	Data = np.float32(channels)
	Data = Data.transpose(1,0,2,3)


	Data = np.true_divide(Data, max_value)
	print(Data.shape)


	save_path = os.path.join(save_dir_path , "rbc_{}/noramlized_by_max/".format(number_of_timesteps))

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	np.save( str(save_path) + "{}.npy".format(start_time), Data)




	
	
	
	
	
	

#     save_path = os.path.join(save_dir_path , "rbc_{}/max/".format(number_of_timesteps))

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)


#     Data={'ux_data':ux_data, 'uy_data': uy_data}
#     np.save( str(save_path) + "original_data_{}.npy".format(start_time), Data)




