import numpy as np
import os
from glob import glob
from ops import lerp

 
import numpy as np
import re
import tensorflow as tf
# import tensorflow_datasets as tfds


import tensorflow.contrib.slim as slim
import cv2




def read_npy_file(item, res):
	# NCHW ---> NHWC
	data = np.load(item.decode()).transpose(0,2,3,1)
	return data.astype(np.float32)



def load_from_numpy(dataset_name):
	filelist =  sorted(glob("/global/cscratch1/sd/rgupta2/backup/netcdf_256_resolution_2_channels/rbc_500/*.npy"))[:2]
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

def save_images(images, size, image_path):
	# return imsave(inverse_transform(images), size, image_path)
	return imsave(images, size, image_path)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	c = images.shape[3]
	img = np.zeros((h * size[0], w * size[1], c))
	for idx, image in enumerate(images):
		i = idx % size[1]
		j = idx // size[1]
		img[h*j:h*(j+1), w*i:w*(i+1), :] = image

	return img

def imsave(images, size, path):
	# return scipy.misc.imsave(path, merge(images, size))

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

