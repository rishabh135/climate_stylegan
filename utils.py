
from __future__ import print_function
import os, sys
from PIL import Image
import os
from glob import glob
from ops import lerp

import numpy as np
import re
import tensorflow as tf
# import tensorflow_datasets as tfds
import scipy

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 4})

import tensorflow.contrib.slim as slim
import cv2

import radialProfile

def _resize_image(image, target):
	return cv2.resize(image, dsize=(target, target), interpolation=cv2.INTER_AREA)


# import kornia
# def divergence(preds):
# 	#preds: batch_size*output_steps*2*H*W
# 	preds_u = preds[:,:,0]
# 	preds_v = preds[:,:,1]
# 	u = torch.from_numpy(preds_u).float().to(device)
# 	v = torch.from_numpy(preds_v).float().to(device)
# 	#Sobolev gradients
# 	field_grad = kornia.filters.SpatialGradient()
# 	u_x = field_grad(u)[:,:,0]
# 	v_y = field_grad(v)[:,:,1]
# 	div = torch.mean(torch.abs(v_y + u_x), dim = (0,2,3)).cpu().data.numpy()
# 	return div


def npy_header_offset(npy_path):
	with open(str(npy_path), 'rb') as f:
		if f.read(6) != b'\x93NUMPY':
			raise ValueError('Invalid NPY file.')
		version_major, version_minor = f.read(2)
		if version_major == 1:
			header_len_size = 2
		elif version_major == 2:
			header_len_size = 4
		else:
			raise ValueError('Unknown NPY file version {}.{}.'.format(version_major, version_minor))
		header_len = sum(b << (8 * i) for i, b in enumerate(f.read(header_len_size)))
		header = f.read(header_len)
		if not header.endswith(b'\n'):
			raise ValueError('Invalid NPY file.')
		return f.tell()


def read_npy_file(item, res):
	# NCHW ---> NHWC
	data = np.load(item.decode()).transpose(0,2,3,1)
	image = [_resize_image(image=i, target=res) for i in data]
	data= np.stack(image, axis=0)
	return data.astype(np.float32)
	# print("data: ",data.shape)
	# print("inside read_npy_file first line : ", type(data))
	# data = data.astype(np.float32)
	# print("data: ",data.shape)
	# print("res: ", res)
	# data =  cv2.resize(data, dsize=(res, res), interpolation=cv2.INTER_AREA)
	# print("inside read_npy_file : ", type(data))
	# print("data: ",data.shape)



def create_from_numpy(tfrecord_dir, numpy_filename, shuffle, res, channels):

	# data = [np.load(file.decode()).transpose(0,2,3,1) for file in filelist]
	# return data

	# reading in NCHW format
	all_arr = np.load(numpy_filename, mmap_mode='r')

	img = all_arr[0]
	resolution = img.shape[1]
	
	# channels = 3 if len(img.shape) == 3 else 1
	
	if img.shape[0] != resolution:
		error('Input images must have the same width and height')
	if resolution != 2 ** int(np.floor(np.log2(resolution))):
		error('Input image resolution must be a power-of-two')
	# if channels not in [1, 3]:
	# 	error('Input images must be stored as RGB or grayscale')

	# is_HWC = (channels == 3) and (img.shape[2] == 3)
	is_HWC = False
	with TFRecordExporter(tfrecord_dir, all_arr.shape[0]) as tfr:
		order = tfr.choose_shuffled_order() if shuffle else np.arange(all_arr.shape[0])
		for idx in range(order.size):
			img = all_arr[order[idx]]
			if channels == 1:
				img = img[np.newaxis, :, :] # HW => CHW
			elif is_HWC:
				img = img.transpose([2, 0, 1]) # HWC => CHW
			img = _resize_image(image=img, target=res)
			img = np.asarray(img) 
			tfr.add_image(img)




def load_from_numpy(dataset_name, dataset_location):
	# filelist = [] 

	"""
	filelist = hacky way of selecting all numpy files with basename starting with integers
	"""
									
	filelist = sorted(glob("{}/*.npy".format(dataset_location)))[:-1]
	# for file in tmp_list:
	# 	if re.match("^\d", file):
	# 		filelist.append(file)
	# print(filelist)
	return filelist



class ImageData:

	def __init__(self, img_size):
		self.img_size = img_size
		self.channels = 3

	def image_processing(self, filename):
		x = tf.read_file(filename)
		img = tf.image.decode_jpeg(x, channels=self.channels, dct_method='INTEGER_ACCURATE')
		img = preprocess_fit_train_image(img, self.img_size)

		return img




# class RBCData:

# 	def __init__(self, img_size, num_channels):
# 		self.img_size = img_size
# 		self.channels = num_channels

# 	def image_processing(self, filename):
# 		data = np.load(filename.decode()).transpose(0,2,3,1)
# 		data = tf.image.resize(data, size=[self.img_size, self.img_size], method=tf.image.ResizeMethod.BILINEAR)
# 		# print("inside read_npy_file : ", type(data))
# 		return data



def adjust_dynamic_range(images):
	drange_in = [0.0, 255.0]
	drange_out = [-1.0, 1.0]
	scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
	bias = drange_out[0] - drange_in[0] * scale
	images = images * scale + bias
	return images


def random_flip_left_right(images):
	s = tf.shape(images)
	mask = tf.random_uniform([1, 1, 1], 0.0, 1.0)
	mask = tf.tile(mask, [s[0], s[1], s[2]]) # [h, w, c]
	images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[1]))
	return images


def smooth_crossfade(images, alpha):
	s = tf.shape(images)
	y = tf.reshape(images, [-1, s[1] // 2, 2, s[2] // 2, 2, s[3]])
	y = tf.reduce_mean(y, axis=[2, 4], keepdims=True)
	y = tf.tile(y, [1, 1, 2, 1, 2, 1])
	y = tf.reshape(y, [-1, s[1], s[2], s[3]])
	images = lerp(images, y, alpha)
	return images

def preprocess_fit_train_image(images, res):
	images = tf.image.resize(images, size=[res, res], method=tf.image.ResizeMethod.BILINEAR)
	images = adjust_dynamic_range(images)
	images = random_flip_left_right(images)

	return images

def load_data(dataset_name) :
	x = glob(os.path.join("/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset", dataset_name, '*.*'))

	return x

def save_images(images, size, image_path, rbc_data, current_res):
	# return imsave(inverse_transform(images), size, image_path)
	return imsave(images, size, image_path, rbc_data, current_res)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	c = images.shape[3]
	img = np.zeros((h * size[0], w * size[1], c))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[h*j:h*(j+1), w*i:w*(i+1), :] = image

	return img

def tke2spectrum(tke, res, dash=None):
	"""Convert TKE field to spectrum"""
	sp = np.fft.fft2(tke.reshape(res,res))
	sp = np.fft.fftshift(sp)
	sp = np.real(sp*np.conjugate(sp))
	sp1D = radialProfile.azimuthalAverage(sp)
	return sp1D


	# if dash == None:
	# 	p1, = 
	# else:
	# 	p1, = plt.plot(sp1D, ls, linewidth=2, dashes=dash)
	# return p1




def get_turbulent_kinetic_energy(images, number_of_images, res, save_path, dataset_location, experiment, real_images):



	load_path = os.path.join(dataset_location + "tke_average_energies.npy")


	my_dict_back = np.load(load_path, allow_pickle=True)
	
	ux_average_over_time = my_dict_back.item()["{}_ux".format(res)]
	uy_average_over_time = my_dict_back.item()["{}_uy".format(res)]

	fig2=plt.figure(figsize=(10,3))	
	

	idx = 1
	for ii , image in enumerate(images):


		"""

		calculating turbulent kinetic energy over all timesteps from 10,000 to 15,0000 as training steps
		

			E_x(x, y) = U_x(x, y) - \intgeral_time{U_x(x, y)}
			E_y(x, y) = U_y(x, y) - \intgeral_time{U_y(x, y)}
			TKE**2 = E_x**2 + E_y**2


		"""


		fig2.add_subplot( number_of_images, 4, idx )
		idx += 1
		ux_current, uy_current = image[:,:,0], image[:,:,1]		
		tke = np.sqrt((ux_current - ux_average_over_time)**2 + (uy_current - uy_average_over_time)**2)
		plt.imshow(tke)		


		fig2.add_subplot( number_of_images, 4, idx )
		idx += 1
		ux_real, uy_real = real_images[ ii ,:,:,0], real_images[ ii, :,:,1]
		
		tke_real = np.sqrt((ux_real - ux_average_over_time)**2 + (uy_real - uy_average_over_time)**2)
		plt.imshow(tke_real)	




		fig2.add_subplot( number_of_images, 4, idx)
		idx += 1
		plt.hist([tke.flatten(), tke_real.flatten()], bins='fd')




		ax = fig2.add_subplot(  number_of_images, 4, idx)

		idx += 1
		sp1D = tke2spectrum(tke,res)
		sp1D_real = tke2spectrum(tke_real,res)
		ax.plot(sp1D_real, '-.g',  linewidth=0.25)
		ax.plot(sp1D, ':r', linewidth=0.25)
		ax.set_yscale("log")



	plt.subplots_adjust( hspace=0, wspace=0)
	fig2.tight_layout()

	indx = re.findall('\d+', save_path)[-1]
	path = os.path.join( os.path.dirname(save_path), "tke_plots_for_res_{}_idx_{}.jpg".format(res, indx)) 	
	plt.savefig(path, dpi = 400)

	experiment.log_figure(figure=plt,  figure_name="Turbulent Kinetic energy at res {} and index {} ".format(res, indx))
	





def visualize_each_image_test(images, size, path, experiemnt, dataset_location = None, num_images_to_be_shown=4):
	h, w = images.shape[1], images.shape[2]
	c = images.shape[3]
		
	# number_of_images = min(images.shape[0], num_images_to_be_shown)


	fig=plt.figure()				
	idx=1
	for  ii , image in enumerate(images):

		plt.clf()
		# image = post_process_generator_output(image)
		ux_data_plot, uy_data_plot = image[:,:,0], image[:,:,1]

		ax = fig.add_subplot( 1, 2, idx)
		idx += 1
		plt.imshow(ux_data_plot)
		plt.xticks([])
		plt.yticks([])
		

		ax = fig.add_subplot(  1, 2, idx)
		idx += 1
		plt.hist(total_preds.flatten(), bins='fd',histtype='step', label="{}".format(counter), density=True)
		plt.imshow(uy_data_plot)
		plt.xticks([])
		plt.yticks([])
		

	

		plt.subplots_adjust( hspace=0, wspace=0)
		fig.tight_layout()
		plt.savefig(path, dpi = 400)
	
		experiment.log_figure(figure=plt,  figure_name="test image at size {} for ii {}".format(size, ii))







def calculate_divergence_tf(generated_data, real_data):

	"""
	generated_data shape is  64*NH*2


	"""
	
	# randn = np.random.normal(loc=0, scale=1,size=(100,4,4,2))
	# generated_data = tf.convert_to_tensor(generated_data,dtype=tf.float32)


	ux_x = tf.image.sobel_edges(generated_data[:,:,:,:1])[:,:,:,:,1]
	uy_y = tf.image.sobel_edges(generated_data[:,:,:,-1:])[:,:,:,:,0]
	tmp =  (ux_x + uy_y)[:,:,:,0]
	fake_values = tf.reduce_mean(tmp, axis=[1,2])



	ux_x = tf.image.sobel_edges(real_data[:,:,:,:1])[:,:,:,:,1]
	uy_y = tf.image.sobel_edges(real_data[:,:,:,-1:])[:,:,:,:,0]
	tmp =  (ux_x + uy_y)[:,:,:,0]
	real_values = tf.reduce_mean(tmp, axis=[1,2])


	return fake_values, real_values






def calculate_divergence(generated_data, path, experiment, real_images):

	"""
	generated_data shape is  64*NH*2


	"""
	
	# randn = np.random.normal(loc=0, scale=1,size=(100,4,4,2))
	generated_data = tf.convert_to_tensor(generated_data,dtype=tf.float32)


	ux_x = tf.image.sobel_edges(generated_data[:,:,:,:1])[:,:,:,:,1]
	uy_y = tf.image.sobel_edges(generated_data[:,:,:,-1:])[:,:,:,:,0]
	tmp =  (ux_x + uy_y)[:,:,:,0]
	fake_values = tf.reduce_mean(tmp, axis=[1,2])






	real_data = tf.convert_to_tensor(real_images,dtype=tf.float32)

	ux_x = tf.image.sobel_edges(real_data[:,:,:,:1])[:,:,:,:,1]
	uy_y = tf.image.sobel_edges(real_data[:,:,:,-1:])[:,:,:,:,0]
	tmp =  (ux_x + uy_y)[:,:,:,0]
	real_values = tf.reduce_mean(tmp, axis=[1,2])


	fig=plt.figure()
	plt.hist([fake_values.flatten(), real_values.flatten() ], bins='fd', histtype='step', density=True, color=["green","red"])	

	experiment.log_figure(figure=plt,  figure_name=" divergence plots with {} images".format(real_images.shape[0]))




	# ux_x = tf.image.sobel_edges(generated_data[:,:,:,:1])[:,:,:,:,1]
	# uy_y = tf.image.sobel_edges(generated_data[:,:,:,-1:])[:,:,:,:,0]
	# values = tf.reduce_mean((ux_x + uy_y)[:,:,:,0], axis=0)
	# print("shape of ux_x {}, values {}".format(tf.shape(ux_x), tf.shape(values)))

	# number_of_images = min(images.shape[0], num_images_to_be_shown)


	# fig=plt.figure()				
	# idx=1


	# ux_real , uy_real = real_images[ 0, :, :, 0], real_images[ 0, :, :, 1]

	# fig.add_subplot( 2, 2, idx)
	# idx += 1
	# plt.imshow(ux_data[0])
	# plt.xticks([])
	# plt.yticks([])
	

	# fig.add_subplot( 2, 2, idx)
	# idx += 1
	# plt.imshow(ux_real)
	# plt.xticks([])
	# plt.yticks([])



	# fig.add_subplot( 2, 2, idx)
	# idx += 1
	# plt.imshow(uy_data[0])
	# plt.xticks([])
	# plt.yticks([])
	

	# fig.add_subplot( 2, 2, idx)
	# idx += 1
	# plt.imshow(uy_real)
	# plt.xticks([])
	# plt.yticks([])

	# plt.subplots_adjust( hspace=0, wspace=0)
	# fig.tight_layout()
	# plt.savefig(path, dpi = 400)

	# experiment.log_figure(figure=plt,  figure_name=" plotting original and generated plots at {}".format(res))
















def plot_generated_velocity(ux_data, uy_data, path, experiment, real_images):



	res = ux_data.shape[-1]
	# number_of_images = min(images.shape[0], num_images_to_be_shown)


	fig=plt.figure()				
	idx=1


	ux_real , uy_real = real_images[ 0, :, :, 0], real_images[ 0, :, :, 1]

	fig.add_subplot( 2, 2, idx)
	idx += 1
	plt.imshow(ux_data[0])
	plt.xticks([])
	plt.yticks([])
	

	fig.add_subplot( 2, 2, idx)
	idx += 1
	plt.imshow(ux_real)
	plt.xticks([])
	plt.yticks([])



	fig.add_subplot( 2, 2, idx)
	idx += 1
	plt.imshow(uy_data[0])
	plt.xticks([])
	plt.yticks([])
	

	fig.add_subplot( 2, 2, idx)
	idx += 1
	plt.imshow(uy_real)
	plt.xticks([])
	plt.yticks([])

	plt.subplots_adjust( hspace=0, wspace=0)
	fig.tight_layout()
	plt.savefig(path, dpi = 400)

	experiment.log_figure(figure=plt,  figure_name=" plotting original and generated plots at {}".format(res))




def plot_velocity_hist(ux_data, uy_data, path, experiment, real_images, dataset_location = None, num_images_to_be_shown=4):


	res = ux_data.shape[-1]
	# number_of_images = min(images.shape[0], num_images_to_be_shown)


	fig=plt.figure()				
	idx=1

	ux_real , uy_real = real_images[ :, :, :, 0], real_images[ :, :, :, 1]	


	ax = fig.add_subplot( 1, 2, idx)
	idx += 1
	plt.hist([ux_data.flatten(), ux_real.flatten() ], bins='fd',histtype='step', density=True, color=["green","red"])
	# plt.xticks([])
	# plt.yticks([])
	

	ax = fig.add_subplot(  1, 2, idx)
	idx += 1
	plt.hist([uy_data.flatten(),uy_real.flatten()] , bins='fd',histtype='step', density=True, color=["green","red"])
	
	# plt.xticks([])
	# plt.yticks([])
	



	plt.subplots_adjust( hspace=0, wspace=0)
	fig.tight_layout()
	plt.savefig(path, dpi = 400)

	experiment.log_figure(figure=plt,  figure_name=" plotting ux and uy histogram at res : {}".format(res))













def imsave(images, size, path, rbc_data, current_res, dataset_location="", experiment=None, num_images_to_be_shown=4, real_images=None ):
	# return scipy.misc.imsave(path, merge(images, size))

	if(rbc_data):

		h, w = images.shape[1], images.shape[2]
		c = images.shape[3]
		
		number_of_images = min(images.shape[0], num_images_to_be_shown)




		# get_turbulent_kinetic_energy(images[: number_of_images, :,:,:], number_of_images, current_res, path, dataset_location, experiment, real_images)


		fig=plt.figure(figsize=(5,5))				
		idx=1

		for  ii , image in enumerate(images):

			if(idx > number_of_images*2):
				break;

			# image = post_process_generator_output(image)
			ux_data_plot, uy_data_plot = image[:,:,0], image[:,:,1]
			ux_real, uy_real = real_images[ ii ,:,:,0], real_images[ ii, :,:,1]


			fig.add_subplot(  number_of_images, 6, idx)
			idx += 1
			plt.imshow(ux_data_plot)
			plt.xticks([])
			plt.yticks([])
			

			fig.add_subplot(   number_of_images, 6, idx)
			
			idx += 1
			plt.imshow(ux_real)
			plt.xticks([])
			plt.yticks([])
			
			
			
			fig.add_subplot(  number_of_images, 6, idx)
			idx += 1
			plt.imshow(uy_data_plot)
			plt.xticks([])
			plt.yticks([])


			fig.add_subplot(  number_of_images, 6, idx)
			idx += 1
			plt.imshow(uy_real)
			plt.xticks([])
			plt.yticks([])


			fig.add_subplot( number_of_images, 6, idx)
			idx += 1
			plt.hist([ux_data_plot.flatten(), ux_real.flatten()], bins='fd')
		

			fig.add_subplot( number_of_images, 6, idx)
			idx += 1
			plt.hist([uy_data_plot.flatten(), uy_real.flatten()], bins='fd')



		plt.subplots_adjust( hspace=0, wspace=0)
		fig.tight_layout()
		plt.savefig(path, dpi = 400)
		return plt	
	
	else:
		images = merge(images, size)
		images = post_process_generator_output(images)
		images = cv2.cvtColor(images.astype('uint8'), cv2.COLOR_RGB2BGR)
		cv2.imwrite(path, images)






def inverse_transform(images):
	return (images+1.)/2.

def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir

def show_all_variables():
	model_vars = tf.trainable_variables()
	slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
	return x.lower() in ('true')

def get_checkpoint_res(checkpoint_counter, batch_sizes, iteration, start_res, end_res, gpu_num, end_iteration, do_trans) :
	batch_sizes_key = list(batch_sizes.keys())

	start_index = batch_sizes_key.index(start_res)

	iteration_per_res = []

	for res, bs in batch_sizes.items() :

		if do_trans[res] :
			if res == end_res :
				iteration_per_res.append(end_iteration // (bs * gpu_num))
			else :
				iteration_per_res.append(iteration // (bs * gpu_num))
		else :
			iteration_per_res.append((iteration // 2) // (bs * gpu_num))

	iteration_per_res = iteration_per_res[start_index:]

	for i in range(len(iteration_per_res)) :

		checkpoint_counter = checkpoint_counter - iteration_per_res[i]

		if checkpoint_counter < 1 :
			return i+start_index

def post_process_generator_output(generator_output):

	drange_min, drange_max = -1.0, 1.0
	scale = 255.0 / (drange_max - drange_min)

	scaled_image = generator_output * scale + (0.5 - drange_min * scale)
	scaled_image = np.clip(scaled_image, 0, 255)

	return scaled_image



################################################################################


# Source : https://github.com/MustafaMustafa/stylegan/commit/92b813dd1280aaf93138279bccbe038eb816db61

def create_from_numpy(tfrecord_dir, numpy_filename, shuffle):

	all_arr = np.load(numpy_filename, mmap_mode='r')

	img = all_arr[0]
	resolution = img.shape[1]
	channels = 3 if len(img.shape) == 3 else 1
	if img.shape[0] != resolution:
		error('Input images must have the same width and height')
	if resolution != 2 ** int(np.floor(np.log2(resolution))):
		error('Input image resolution must be a power-of-two')
	if channels not in [1, 3]:
		error('Input images must be stored as RGB or grayscale')

	is_HWC = (channels == 3) and (img.shape[2] == 3)
	with TFRecordExporter(tfrecord_dir, all_arr.shape[0]) as tfr:
		order = tfr.choose_shuffled_order() if shuffle else np.arange(all_arr.shape[0])
		for idx in range(order.size):
			img = all_arr[order[idx]]
			if channels == 1:
				img = img[np.newaxis, :, :] # HW => CHW
			elif is_HWC:
				img = img.transpose([2, 0, 1]) # HWC => CHW
			tfr.add_image(img)

#----------------------------------------------------------------------------

