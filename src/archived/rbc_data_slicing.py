
import gc
import os
import time
import numpy as np
import time
# from netCDF4 import Dataset
import matplotlib.pyplot as plt
import xarray as xr



file_path = "/project/projectdirs/dasrepo/mustafa/datasets/Rayleigh_Benard/result_rb_2d__Ra_2.5e8__Pr_0.71__maxMach_0.1__t_D_max_diffusive_scaling__0.4.nc"






t = time.time()
ds = xr.open_dataset(file_path, decode_times=False  )
print("Time takes : ", time.time()-t)







save_dir_path = "/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/rbc_500/max/"

number_of_timesteps = 500
st_time = 10000
end_time = 20000
image_size = 256



tt = time.time()








for start_time in range(st_time, end_time, number_of_timesteps):
	ux_data = ds['u_x'][start_time : start_time+ number_of_timesteps].values
	uy_data = ds['u_y'][start_time : start_time+ number_of_timesteps].values
	
	
	print(ux_data.shape)



	fig=plt.figure(figsize=(5, 5))
	idx=1	
	
	ux_std = ux_data.std()
	uy_std = uy_data.std()
	
	u_data_max= max( ux_data.max(), uy_data.max())
	
	print("Ux_data_max = {} and Uy data max = {} and u_data_max {}".format(ux_data.max(), uy_data.max(), u_data_max))
	print("Ux std = {} and Uy std = {}".format(ux_std, uy_std))
	
	

	print("Time taken to read all", time.time() - tt)
	rb_data = []
	
	
	strides =  [x * ux_data.itemsize for x in [ux_data.shape[-1] * ux_data.shape[-2], image_size, ux_data.shape[-1], 1]]
	print("strides : {} ".format(strides))
	u_x = np.lib.stride_tricks.as_strided(ux_data, (number_of_timesteps, 7, image_size, image_size), strides)
	

	
	
	strides =  [x * uy_data.itemsize for x in [uy_data.shape[-1] * uy_data.shape[-2], image_size, uy_data.shape[-1], 1]]
	print("strides : {} ".format(strides))
	u_y = np.lib.stride_tricks.as_strided(uy_data, (number_of_timesteps, 7, image_size, image_size), strides)
	
	
	
	print("u-x shape {}  u_y shape  {} ".format(u_x.shape, u_y.shape))
	
	tmp_plot = u_y.reshape(-1,image_size, image_size)
   

	fig.add_subplot(2, 2, idx)
	idx += 1
	plt.imshow(u_y[0,0])
	plt.xticks([])
	plt.yticks([])
	
	
	fig.add_subplot(2, 2,  idx)
	idx += 1
	plt.imshow(tmp_plot[0])
	plt.xticks([])
	plt.yticks([])
	
	fig.add_subplot(2, 2, idx)
	idx += 1
	plt.imshow(u_y[0,1])
	plt.xticks([])
	plt.yticks([])
	
	
	fig.add_subplot(2, 2,  idx)
	idx += 1
	plt.imshow(tmp_plot[1])
	plt.xticks([])
	plt.yticks([])


		
	
	print("Time taken to complete iteration", time.time() - tt)
	
	plt.savefig(os.path.join(save_dir_path , "uy_sliced_correctly_check.png"), dpi=400)
	gc.collect
	
#     u_x = ux_data.reshape( -1, image_size,image_size)/ux_std
#     u_y = uy_data.reshape( -1, image_size,image_size)/uy_std
#     tmp_diff = tmp_diff_data.reshape(number_of_timesteps, -1, image_size, image_size)
#     press_diff = press_diff_data.reshape(number_of_timesteps, -1, image_size, image_size)
	
	
#     channels = np.array([u_x, u_y])
#     print(channels.shape)
#     Data = np.float32(channels)
#     Data = Data.transpose(1,0,2,3)
# #     Data = Data.reshape( -1,2,image_size, image_size)
#     print(Data.shape)
	
	
#     save_path = os.path.join(dir_path , "rbc_{}/".format(number_of_timesteps))

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     np.save( str(save_path) + "{}.npy".format(start_time), Data)
