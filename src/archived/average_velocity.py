import sys,os,os.path,time
# sys.path.append(os.path.expanduser('/global/u1/r/rgupta2/.local/lib/python3.7/site-packages/'))

# !{sys.executable} -m pip install numpy
# !{sys.executable} -m pip install pims
# !{sys.executable} -m pip install --user xarray


# export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/site-packages"
import numpy as np
import time

import matplotlib.pyplot as plt
import xarray as xr

def calculate_mean_average_velocities(file_path):


  t = time.time()
  ds = xr.open_dataset(file_path, decode_times=False  )
  print("Time takes : ", time.time()-t)

  Data = {}



  st_time = 3500
  end_time = 23630
  res = 256
  number_of_timesteps = 3500


  save_dir_path = "/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/one_seventh/"
  save_path = os.path.join(save_dir_path , "rbc_{}/noramlized_by_max/avg_velocities/".format(number_of_timesteps))



  if (not os.path.exists(save_path)):
    os.makedirs(save_path)



  ux_data = ds['u_x'][st_time : end_time].values[:, :, :res]
  uy_data = ds['u_y'][st_time : end_time].values[:, :, :res]




  ux_data_whole_max = ux_data.max()
  uy_data_whole_max = uy_data.max()
  max_value = max(ux_data_whole_max, uy_data_whole_max)



  ux_data_mean_tke = np.mean(ux_data, axis = 0)
  uy_data_mean_tke = np.mean(uy_data, axis = 0)
  
  print("just to make sure ux_data shape is {} ".format(ux_data_mean_tke.shape))

  print("Time taken to complete iteration", time.time() - t)
  
  Data["ux"] = ux_data_mean_tke
  Data["uy"] = uy_data_mean_tke
  Data["max_value"] = max_value
  np.save(save_path + "average_velocity_at_{}.npy".format(res), Data)
  return True;

  # save_path = os.path.join(  + "".format(res))

"""

IsADirectoryError: [Errno 21] Is a directory: '/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/one_seventh/rbc_3500/noramlized_by_max/avg_velocities/average_velocity_at_256.npy'

"""

    
  # if(not os.path.isfile(save_path)):






file_path = "/project/projectdirs/dasrepo/mustafa/datasets/Rayleigh_Benard/result_rb_2d__Ra_2.5e8__Pr_0.71__maxMach_0.1__t_D_max_diffusive_scaling__0.4.nc"
  
done = calculate_mean_average_velocities(file_path)
print(done)