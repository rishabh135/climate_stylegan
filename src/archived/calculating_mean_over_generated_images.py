
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import time, re, sys, os
import numpy as np
import PIL.Image
from tqdm import tqdm
from glob import glob

directory = "/data0/rgupta2/wo_style_rbc_model_weights/results/generated_images/"



counter = 354218

path = (directory + "generator_{}_images_50_file_*.npy".format(counter))


files = sorted(glob(path))

print("Number of files : {}".format(len(files)))

list_of_generated_images = []

for file in files:
	my_dict_back = np.load(file, allow_pickle=True)	
	generated_images = my_dict_back.item()["generated_images"]
	list_of_generated_images.append(generated_images)
	



	
print("total_number_of_files: {} and shape of each file {}".format(len(list_of_generated_images) ,  list_of_generated_images[0].shape))
   


save_path = "/data0/rgupta2/wo_style_rbc_model_weights/results/average_velocity/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
    
final_array = np.concatenate(list_of_generated_images, axis=0 )

Data = {}
mean_img = final_array.mean(axis=0)
Data["256_ux"] = mean_img[:, : ,0]
Data["256_uy"] = mean_img[:, : ,1]
print("mean img shape {} ".format(mean_img.shape))



np.save(save_path + "generator_{}_average_velocities.npy".format(counter), Data)