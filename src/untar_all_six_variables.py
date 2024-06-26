

import tarfile
import sys,os,os.path,time
# sys.path.append(os.path.expanduser('/global/u1/r/rgupta2/.local/lib/python3.7/site-packages/'))

# export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python2.7/site-packages:/usr/lib/python2.7/site-packages"
import numpy as np
import re
from glob import glob
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
# import xarray as xr



t = time.time()
"""
Simple Question Answering with Subgraph Ranking and Joint-Scoring, NAACL 2019, Wenbo Zhao, Tagyoung Chung, Anuj Goyal, Angeliki Metallinou

Unsupervised Transfer Learning for Spoken Language Understanding in Intelligent Agents, AAAI 2019, Aditya Siddhant, Anuj Goyal, Angeliki Metallinou

Online Embedding Compression for Text Classification Using Low Rank Matrix Factorization, AAAI 2019, Anish Acharya, Rahul Goel, Angeliki Metallinou, Inderjit Dhillon

"""

six_vars = ["pr", "prw", "psl", "ts", "ua850", "va850"]




# input_file_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/download_all_vars/"

# input_files = sorted(glob(input_file_path + "*.tar"))
# print(input_files)
  



def extract_tarfile(channel):
  tarfile_path = "/global/cscratch1/sd/karthik_/CAM5.1_0.25degree/download_all_vars/{}_A3hr_CAM5-1-2-025degree_All-Hist_est1_v1-0_run001.tar".format(channel)

  # tarfile_extract_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/CAM5_tar/tmp_untar/"
  save_dir_path = "/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/all_6_variables_nc_files/{}/".format(channel)
  if not os.path.exists(save_dir_path):
    os.makedirs(save_dir_path)


  t = tarfile.open(tarfile_path, 'r')
  t.extractall(path =save_dir_path )
  t.close()





  # for file in tarfiles:
  #   all_ints = re.findall("\d+", file)
  #   month = all_ints[-1]
    


  #   netfiles_path = tarfile_extract_path+"year_{}_month_{}/*.nc".format(year, month)
  #   net_cdf_files = sorted(glob(netfiles_path))

  #   print("\n files found in the current directory for year {} = {}".format(year, len(net_cdf_files)))
        
  #   for _, netfile in enumerate(net_cdf_files):

  #     tmp_ds = xr.open_dataset(netfile, decode_times=False)
  #     list_of_omega_data.append( np.expand_dims(tmp_ds["OMEGA500"][:, 128:640, 320:832].values, axis=1))




  # tt = time.time()
  # numpy_array_of_omega = np.concatenate(list_of_omega_data, axis=0)


  # print("\n\n****** data from year {}  [:, 128:640, 320:832] shape {}, max {} min {} std {} mean {} \n\n ".format(year, numpy_array_of_omega.shape, numpy_array_of_omega.max(), numpy_array_of_omega.min(), numpy_array_of_omega.std(), numpy_array_of_omega.mean() ))

  # if not os.path.exists(save_dir_path):
  #   os.makedirs(save_dir_path)

  # np.save( str(save_dir_path) + "{}.npy".format(year), numpy_array_of_omega)
  # print("Time taken to complete iteration", time.time() - tt)







for channel in six_vars:
  extract_tarfile(channel)

print("\n\n DONE!!")



"""


****** data from year 2012  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 8.46811580657959 min -11.7531156539917 std 0.19989439845085144 mean -0.000696208153385669 

 
Time taken to complete iteration 5.821779012680054

****** data from year 2013  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.542869567871094 min -14.95754337310791 std 0.2030375748872757 mean -0.001598570728674531 

 
Time taken to complete iteration 6.074418306350708


****** data from year 2014  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 7.317117214202881 min -15.671699523925781 std 0.20072969794273376 mean -0.001538553973659873 

 
Time taken to complete iteration 5.8916425704956055



****** data from year 2015  [:, 128:640, 320:832] shape (2920, 1, 512, 512), max 9.217764854431152 min -12.647174835205078 std 0.20094890892505646 mean -0.0014804755337536335 

 
Time taken to complete iteration 6.013826847076416


 DONE!!



"""
