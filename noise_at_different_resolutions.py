from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
	
from comet_ml import Experiment

# from StyleGAN import StyleGAN

import argparse
from utils import *
# import xarray as xr
from glob import glob
import sys

# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="YC7c0hMcGsJyRRjD98waGBcVa",
# 						project_name="on-celeba", workspace="style-gan")

import time, re, sys
from tqdm import tqdm, trange
from ops import *
from utils import *
import tensorflow
print(tensorflow.__version__)
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import PIL.Image


# import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
# import seaborn.timeseries



# def tsplot(data,**kw):
# 	x = np.arange(data.shape[1])
# 	est = np.mean(data, axis=0)
# 	sd = np.std(data, axis=0)
# 	cis = (est - sd, est + sd)
# 	plt.plot(x,est,**kw)
# 	# plt.fill_between(x,cis[0],cis[1], alpha=0.3, **kw)
# 	plt.margins(x=0)


class StyleGAN(object):

	def __init__(self, sess, args, experiment):

		self.experiment = experiment
		self.phase = args.phase
		self.progressive = args.progressive
		self.model_name = "StyleGAN"
		self.sess = sess
		self.dataset_name = args.dataset
		self.checkpoint_dir = args.checkpoint_dir
		# self.sample_dir = args.sample_dir
		self.result_dir = args.result_dir
		# self.log_dir = args.log_dir

		self.iteration = args.iteration * 10000
		self.max_iteration = args.max_iteration * 10000

		self.batch_size = args.batch_size
		self.img_size = args.img_size

		""" Hyper-parameter"""
		self.start_res = args.start_res
		self.resolutions = resolution_list(self.img_size) # [4, 8, 16, 32, 64, 128, 256, 512, 1024 ...]
		self.featuremaps = featuremap_list(self.img_size) # [512, 512, 512, 512, 256, 128, 64, 32, 16 ...]

		print("resolutions", self.resolutions)
		print("feature maps", self.featuremaps)

		if not self.progressive :
			self.resolutions = [self.resolutions[-1]]
			self.featuremaps = [self.featuremaps[-1]]
			self.start_res = self.resolutions[-1]

		self.gpu_num = args.gpu_num

		self.style_mixing_flag = args.style_mixing_flag
		self.z_dim = 512
		self.w_dim = 512
		self.n_mapping = 8
		self.input_channels = args.input_channels
		self.w_ema_decay = 0.995 # Decay for tracking the moving average of W during training
		self.style_mixing_prob = 0.9 # Probability of mixing styles during training
		self.truncation_psi = 0.7 # Style strength multiplier for the truncation trick
		self.truncation_cutoff = 8 # Number of layers for which to apply the truncation trick

		self.batch_size_base = 4
		self.learning_rate_base = 0.001
		self.num_images_to_be_shown = 4

		## training with trans indicated should we 
		self.train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}

		"""
		
			resolution 4 and 8  --> coarse

			resolution 16 and 32  --> middle
			
			resolution 64, 128 and 256  --> fine
			

		"""

		self.noise_at_training_for = {4: False, 8: False, 16: False, 32: False, 64: False, 128: False, 256: False, 512: True, 1024: True}

		self.batch_sizes = get_batch_sizes(self.gpu_num)

		"""
		if gpu_num == 4:
		x = OrderedDict([(4, 128), (8, 64), (16, 32), (32, 16), (64, 8), (128, 4), (256, 4), (512, 4), (1024, 4)])

		"""
		self.end_iteration = get_end_iteration(self.iteration, self.max_iteration, self.train_with_trans, self.resolutions, self.start_res)
		

		print( "self.end_iteration : ", self.end_iteration)

		self.g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
		self.d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

		self.sn = args.sn

		self.print_freq = {4: 1000, 8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 1000, 256: 1000, 512: 10000, 1024: 10000}

		# self.print_freq = {4: 1000, 8: 500, 16: 500, 32: 500, 64: 100, 128: 100, 256: 10, 512: 10000, 1024: 10000}

		self.save_freq = {4: 1000, 8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 3000, 256: 5000, 512: 10000, 1024: 10000}

		self.print_freq.update((x, y // self.gpu_num) for x, y in self.print_freq.items())
		self.save_freq.update((x, y // self.gpu_num) for x, y in self.save_freq.items())

		self.test_num = args.test_num
		self.seed = args.seed

		self.plotting_histogram_images = 64

		self.rbc_data = self.dataset_name.startswith("rbc")
		
		self.dataset_location = args.dataset_location

		# if(self.rbc_data):
		# 	self.dataset = load_from_numpy(self.dataset_name, self.dataset_location)
		# 	self.dataset_num = 3500 * len(self.dataset)
		# else:
		# 	self.dataset = load_data(dataset_name=self.dataset_name)
		# 	self.dataset_num = len(self.dataset)

		# self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
		# check_folder(self.sample_dir)

		print("##### Information #####")
		print("# dataset : ", self.dataset_name)
		# print("# dataset number : ", self.dataset_num)
		print("# gpu : ", self.gpu_num)
		print("# batch_size in train phase : ", self.batch_sizes)
		print("# batch_size in test phase : ", self.batch_size)

		print("# start resolution : ", self.start_res)
		print("# target resolution : ", self.img_size)
		print("# iteration per resolution : ", self.iteration)

		print("# progressive training : ", self.progressive)
		print("# spectral normalization : ", self.sn)

		print("\n\n")




	@property
	def model_dir(self):

		if self.sn :
			sn = '_sn'
		else :
			sn = ''

		if self.progressive :
			progressive = '_progressive'
		else :
			progressive = ''

		return "{}_{}_{}to{}{}{}".format(self.model_name, self.dataset_name, self.start_res, self.img_size, progressive, sn)


	##################################################################################
	# Generator
	##################################################################################

	def g_mapping(self, z, n_broadcast, reuse=tf.AUTO_REUSE):
		with tf.variable_scope('g_mapping', reuse=reuse):
			# normalize input first
			x = pixel_norm(z)

			# run through mapping network
			for ii in range(self.n_mapping):
				with tf.variable_scope('FC_{:d}'.format(ii)):
					x = fully_connected(x, units=self.w_dim, gain=np.sqrt(2), lrmul=0.01, sn=self.sn)
					x = apply_bias(x, lrmul=0.01)
					x = lrelu(x, alpha=0.2)
			# broadcast to n_layers
			with tf.variable_scope('Broadcast'):
				x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])


		return x

	def g_synthesis(self, w_broadcasted, alpha, resolutions, featuremaps, noise_dict={4: False, 8: True, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}, reuse=tf.AUTO_REUSE):
		with tf.variable_scope('g_synthesis', reuse=reuse):
			coarse_styles, middle_styles, fine_styles = get_style_class(resolutions, featuremaps)
			layer_index = 2

			""" initial layer """
			res = resolutions[0]
			n_f = featuremaps[0]

			
			x = synthesis_const_block(res, w_broadcasted, n_f, noise_dict=noise_dict, sn=self.sn)

			""" remaining layers """
			if self.progressive :


				# if(self.rbc_data):
				# 	images_out = x
				# else:
				
				

				images_out = torgb(x, res, input_channels=self.input_channels, sn=self.sn)				
				coarse_styles.pop(res, None)

				# Coarse style [4 ~ 8]
				# pose, hair, face shape
				for res, n_f in coarse_styles.items():


					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_dict=noise_dict, sn=self.sn)
					img = torgb(x, res, self.input_channels, sn=self.sn)				
					images_out = upscale2d(images_out)
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)
					layer_index += 2


				# print("########################\n\n")
				# Middle style [16 ~ 32]
				# facial features, eye
				for res, n_f in middle_styles.items():
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_dict=noise_dict, sn=self.sn)
					# if(self.rbc_data):
					# 	img = x
					# else:
					
					img = torgb(x, res, self.input_channels,  sn=self.sn)
					
					images_out = upscale2d(images_out)
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)
					layer_index += 2

				# Fine style [64 ~ 1024]
				# color scheme
				for res, n_f in fine_styles.items():
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_dict=noise_dict, sn=self.sn)
					
					# if(self.rbc_data):
					# 	img = x
					# else:
					img = torgb(x, res, self.input_channels, sn=self.sn)
					
					images_out = upscale2d(images_out)
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

					layer_index += 2

			else :
				for res, n_f in zip(resolutions[1:], featuremaps[1:]) :
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, noise_dict=noise_dict, sn=self.sn)

					layer_index += 2

				# if(self.rbc_data):
				# 	images_out = x
				# else:
				
				images_out = torgb(x, resolutions[-1], self.input_channels, sn=self.sn)
	
			return images_out

	def generator(self, z, alpha, target_img_size, is_training=True, reuse=tf.AUTO_REUSE):
		with tf.variable_scope("generator", reuse=reuse):
			resolutions = resolution_list(target_img_size)
			featuremaps = featuremap_list(target_img_size)

			w_avg = tf.get_variable('w_avg', shape=[self.w_dim],
									dtype=tf.float32, initializer=tf.initializers.zeros(),
									trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

			""" mapping layers """
			n_broadcast = len(resolutions) * 2
			w_broadcasted = self.g_mapping(z, n_broadcast)


			if is_training:
				""" apply regularization techniques on training """
				# update moving average of w
				w_broadcasted = self.update_moving_average_of_w(w_broadcasted, w_avg)

				# perform style mixing regularization
				w_broadcasted = self.style_mixing_regularization(z, w_broadcasted, n_broadcast, resolutions, self.style_mixing_flag)


			else :
				""" apply truncation trick on evaluation """
				w_broadcasted = self.truncation_trick(n_broadcast, w_broadcasted, w_avg, self.truncation_psi)

			""" synthesis layers """


			x = self.g_synthesis(w_broadcasted, alpha, resolutions, featuremaps)


			return x

	##################################################################################
	# Discriminator
	##################################################################################

	def discriminator(self, x_init, alpha, target_img_size, reuse=tf.AUTO_REUSE):

		# print("tf.shape of x_init: ", tf.keras.backend.eval(tf.shape(x_init)))	 
		with tf.variable_scope("discriminator", reuse=reuse):
			resolutions = resolution_list(target_img_size)
			featuremaps = featuremap_list(target_img_size)

			r_resolutions = resolutions[::-1]
			r_featuremaps = featuremaps[::-1]

			
			# """ set inputs """
			# if(self.rbc_data):
			# 	x = x_init
			# else:
				
			x = fromrgb(x_init, r_resolutions[0], r_featuremaps[0], self.sn)
				# print("scope of x after fromrgb: ", x.name)	
				# print("shape of x after fromrgb: ", tf.keras.backend.eval(tf.shape(x)))

			""" stack discriminator blocks """
			for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
				res_next = r_resolutions[index + 1]
				n_f_next = r_featuremaps[index + 1]

				x = discriminator_block(x, res, n_f, n_f_next, self.sn)

				if self.progressive :
					x_init = downscale2d(x_init)

					# if(self.rbc_data):
					# 	y = x_init		
					# else:
					
					y = fromrgb(x_init, res_next, n_f_next, self.sn)
					x = smooth_transition(y, x, res, r_resolutions[0], alpha)


			""" last block """
			res = r_resolutions[-1]
			n_f = r_featuremaps[-1]

			logit = discriminator_last_block(x, res, n_f, n_f, self.sn)
					
			return logit

	##################################################################################
	# Technical skills
	##################################################################################

	def update_moving_average_of_w(self, w_broadcasted, w_avg):
		with tf.variable_scope('WAvg'):
			batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
			update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, self.w_ema_decay))

			with tf.control_dependencies([update_op]):
				w_broadcasted = tf.identity(w_broadcasted)

		return w_broadcasted

	def style_mixing_regularization(self, z, w_broadcasted, n_broadcast, resolutions, style_mixing_flag):
		with tf.name_scope('style_mix'):

			if(style_mixing_flag):
				z2 = tf.random_normal(tf.shape(z), dtype=tf.float32)
				w_broadcasted2 = self.g_mapping(z2, n_broadcast)
			else:
				w_broadcasted2 = w_broadcasted


			layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
			last_layer_index = (len(resolutions)) * 2

			mixing_cutoff = tf.cond(tf.random_uniform([], 0.0, 1.0) < self.style_mixing_prob,
				lambda: tf.random_uniform([], 1, last_layer_index, dtype=tf.int32),
				lambda: tf.constant(last_layer_index, dtype=tf.int32))

			w_broadcasted = tf.where(tf.broadcast_to(layer_indices < mixing_cutoff, tf.shape(w_broadcasted)),
									 w_broadcasted,
									 w_broadcasted2)

		return w_broadcasted

	def truncation_trick(self, n_broadcast, w_broadcasted, w_avg, truncation_psi):
		with tf.variable_scope('truncation'):
			layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
			ones = np.ones(layer_indices.shape, dtype=np.float32)
			coefs = tf.where(layer_indices < self.truncation_cutoff, truncation_psi * ones, ones)
			w_broadcasted = lerp(w_avg, w_broadcasted, coefs)

		return w_broadcasted

	##################################################################################
	# Model
	##################################################################################

	def build_model(self):
		test_z = tf.random_normal(shape=[self.batch_size, self.z_dim])
		alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
		self.fake_images = self.generator(test_z, alpha=alpha, target_img_size=self.img_size, is_training=False)

	def load(self, checkpoint_dir, counter):
		
		# checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		
		print(" [*] Reading checkpoints from  {} with counter {} ".format(checkpoint_dir, counter))
		if(counter == 0):
				states = tf.train.get_checkpoint_state(checkpoint_dir)
				checkpoint_paths = states.all_model_checkpoint_paths
				load_model_path = checkpoint_paths[-1]

		else:
			load_model_path =  os.path.join(checkpoint_dir, "StyleGAN.model-{}".format(counter))



		try:
			self.saver.restore(self.sess, load_model_path)
			print(" [*] Success to read {}".format(load_model_path))
			# counter = int(load_model_path.split('-')[-1])
			return True, counter
		
		except:
			print(" [*] Failed to find a checkpoint")
			return False, -1







	def draw_style_mixing_figure(self, generated_data_location, real_data_location):
		tf.global_variables_initializer().run()
		self.saver = tf.train.Saver()


		"""
		Flag to determine whether to plot spectral plots for every row or velocities

		"""
		plot_spectral = True
		noise_times = 5		


		noise_variations = [ {4: False, 8: False, 16: False, 32: False, 64: False, 128: False, 256: False, 512: True, 1024: True},  {4: False, 8: False, 16: False, 32: False, 64: True, 128: True, 256: True, 512: True, 1024: True}, {4: True, 8: True, 16: False, 32: False, 64: False, 128: False, 256: False, 512: True, 1024: True}
		]



		result_dir = os.path.join(self.result_dir, 'noise_mixing_figure/')
		check_folder(result_dir)


		load_path = os.path.join(self.result_dir , generated_data_location[0])
		my_dict_back = np.load(load_path, allow_pickle=True)

		# using re.findall() 
		# getting numbers from string  
		regex_for_numbers = re.findall(r'\d+', load_path ) 
		res = list(map(int, regex_for_numbers)) 
		

		counter = res[-3]
		print("Loaded correct checkpoint with counter {}".format(counter))

		could_load, checkpoint_counter = self.load(self.checkpoint_dir, counter)
		
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")





		real_images = np.load(real_data_location)[:50]

		# generated_images = my_dict_back.item()["generated_images"][:11]


		total_seeds = my_dict_back.item()["seeds"][:11]


		src_seeds = total_seeds[:2]
		# src_seeds = [ x for x in src_seeds for i in range(noise_times) ] 
		# dst_seeds = total_seeds[5:]
		# src_seeds = [604, 8440, 7613, 6978, 3004]
		# dst_seeds = [1336, 6968, 607, 728, 7036, 9010]

		resolutions = resolution_list(self.img_size)
		featuremaps = featuremap_list(self.img_size)
		n_broadcast = len(resolutions) * 2

		# style_ranges = [range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, n_broadcast)]
		# print("style ranged originally {} ".format(style_ranges))
		alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
		
		if self.seed :
			src_latents = tf.cast(np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.z_dim]) for seed in src_seeds), axis=0), tf.float32)
			# dst_latents = tf.cast(np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.z_dim]) for seed in dst_seeds), axis=0), tf.float32)

		else :
			src_latents = tf.cast(np.random.normal(size=[len(src_seeds), self.z_dim]), tf.float32)
			# dst_latents = tf.cast(np.random.normal(size=[len(dst_seeds), self.z_dim]), tf.float32)



		# print("\n\n\n")
		# print("size of src latent {} and that of dst_latents {} ".format(src_latents.shape, dst_latents.shape))


		with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) :

			src_dlatents_original = self.g_mapping(src_latents, n_broadcast)
			# dst_dlatents = self.g_mapping(dst_latents, n_broadcast)


			# print("size of src dlatent {} and that of dst_dlatents {} ".format(src_dlatents.shape, dst_dlatents.shape))


			# dlatent_avg = tf.get_variable('w_avg')

			# src_dlatents = self.truncation_trick(n_broadcast, src_dlatents, dlatent_avg, self.truncation_psi)
			# dst_dlatents = self.truncation_trick(n_broadcast, dst_dlatents, dlatent_avg, self.truncation_psi)

			src_images = self.sess.run(self.g_synthesis(src_dlatents_original, alpha, resolutions, featuremaps, noise_dict = noise_variations[0]))
			# column_images = self.sess.run(self.g_synthesis(src_dlatents, alpha, resolutions, featuremaps))
			# dst_images = self.sess.run(self.g_synthesis(dst_dlatents, alpha, resolutions, featuremaps))

			# for i in range(len(src_images)):
			# 	src_images[i] = post_process_generator_output(src_images[i])

			# for i in range(len(dst_images)):
			# 	dst_images[i] = post_process_generator_output(dst_images[i])

			src_dlatents = self.sess.run(src_dlatents_original)
			# dst_dlatents = self.sess.run(dst_dlatents)

			# canvas = PIL.Image.new('RGB', (self.img_size * (len(src_seeds) + 1), self.img_size * (len(dst_seeds) + 1)), 'white')
			# print("sess.run size {} and that of dst_dlatents after {} ".format(src_dlatents.shape, dst_dlatents.shape))


			print("\n\n\n")
			

				
			fig, axs = plt.subplots( len(noise_variations)+1, len(src_seeds), figsize=(8,20))
			idx = 0


			"""

			plot an empty image for future use

			"""


			# plot_tke(generated_data, , res, dataset_location)

			# ax = fig.add_subplot( len(src_seeds), len(noise_variations),  idx)
			# idx += 1
			# ax.imshow(src_images[0,:,:,0])
			# plt.title('src_image_{}'.format(col), fontsize='xx-small')
			# ax.xticks([])
			# ax.yticks([])
			# ax.set_visible(False)



			for col, src_image in enumerate(list(src_images)):
				# fig.add_subplot(  len(noise_variations)+1, len(src_seeds),  idx)
				if(plot_spectral):
					# generated_images = my_dict_back.item()["generated_images"][:11]
					sp1D_gen, sp1D_real = plot_tke(src_image, real_images, self.img_size, real_data_location)
					axs[idx, col].plot(sp1D_gen/sp1D_real, '-r')
					plt.yscale("log")
				# plt.imshow(src_image[ :, : , 0])
				
				plt.title('original_image_{}'.format(col), fontsize='large')
				# plt.xticks([])
				# plt.yticks([])		
				# canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * self.img_size, 0))

			list_print = ["no noise at any level", "noise only at finer levels 64 and above", "noise only at 8"]

			for row, noise_v in tqdm(enumerate(noise_variations)):
				
				collection_of_radial_profiles = [[] for i in range(2)]
				for i in range(noise_times):
					tmp_images = self.sess.run(self.g_synthesis(src_dlatents_original, alpha, resolutions, featuremaps, noise_dict=noise_v))
					for idx, img in enumerate(list(tmp_images)):
						sp1D_gen, sp1D_real = plot_tke(img, real_images, self.img_size, real_data_location)
						collection_of_radial_profiles[idx].append(sp1D_gen)

				row_images = []

				for i in range(2):
					row_images.append(np.vstack(collection_of_radial_profiles[i]))
					# print("\n\n*********row_image shape {} ".format(row_images[i].shape))
				
				idx += 1
				for col, image in enumerate(row_images):
					# print("*******\n\n row image is of shape {}\n".format(image.shape))
					# fig.add_subplot(  len(noise_variations)+1, len(src_seeds),  idx)
					
					if(plot_spectral):
						
						# x = np.arange(image.shape[1])
						# est = np.mean(image, axis=0)
						# sd = np.std(image, axis=0)
						# cis = (est - sd, est + sd)
						# print("inside plot spectral with xshape {} est shape {} and  *****".format(x.shape, est.shape))
						axs[idx, col].plot(image[0], '.g', linewidth=0.5)
						# plt.fill_between(x,cis[0],cis[1], alpha=0.7)
						# plt.margins(x=0)

						# generated_images = my_dict_back.item()["generated_images"][:11]
						# tsplot(image)
						#tsplot(ax2, image)
						# plt.show(sns)
						# plt.plot(sp1D_gen, "-g")
						plt.yscale("log")
						plt.title('{}'.format(list_print[row]), fontsize='large')
					# plt.imshow(image[ :, : , 0])


			plt.subplots_adjust( hspace=0, wspace=0)
			fig.tight_layout()
			plt.savefig( '{}/noise_variations_profile.jpg'.format(result_dir) , dpi = 400)



			# for row, dst_image in enumerate(list(dst_images)):

			# 	# canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * self.img_size))

			# 	fig.add_subplot( len(dst_seeds)+1, len(src_seeds)+1,  idx)
			# 	idx += 1


			# 	if(plot_spectral):
			# 		# generated_images = my_dict_back.item()["generated_images"][:11]
			# 		sp1D_gen, sp1D_real = plot_tke(dst_image, real_images, self.img_size, dataset_location)
			# 		plt.plot(sp1D_gen, "-g")
			# 		plt.yscale("log")



			# 	# plt.imshow(dst_image[ :, : , 0])
			# 	# plt.title('dst_image_{}'.format(row), fontsize='small')
			# 	# plt.xticks([])
			# 	# plt.yticks([])

			# 	row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
			# 	print("src_dlatents : {} ".format(src_dlatents[:, :, 0]))
			# 	print("\n")
			# 	print("style_ranges : {} with size of row_dlatents {} \n  and row_dlatents original : {}".format(style_ranges[row], row_dlatents.shape, row_dlatents[:,:,0]))
			# 	row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]

			# 	print("\n")
			# 	print("row_dlatents after modifying style {} ".format(row_dlatents[:,:,0]))
			# 	print("*************************\n\n")
			# 	row_images = self.sess.run(self.g_synthesis(tf.convert_to_tensor(row_dlatents, tf.float32), alpha, resolutions, featuremaps))

			# 	# for i in range(len(row_images)):
			# 	# 	row_images[i] = post_process_generator_output(row_images[i])

			# 	for col, image in enumerate(list(row_images)):
			# 		fig.add_subplot(  len(dst_seeds)+1, len(src_seeds)+1,  idx)
			# 		idx += 1
			# 		plt.imshow(image[ :, : , 0])
			# 		plt.title('row_image_{}'.format(col), fontsize='small')
			# 		plt.xticks([])
			# 		plt.yticks([])	
			# 		# canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'), ((col + 1) * self.img_size, (row + 1) * self.img_size))










def tke2spectrum(tke, res, dash=None):
	"""Convert TKE field to spectrum"""
	sp = np.fft.fft2(tke.reshape(res,res))
	sp = np.fft.fftshift(sp)
	sp = np.real(sp*np.conjugate(sp))
	sp1D = radialProfile.azimuthalAverage(sp)
	return sp1D








def plot_tke(generated_data, real_images, res, dataset_location):


	ux_data = generated_data[ :, :, 0]
	uy_data = generated_data[ :, :, 1]

	ux_real = real_images[:, 0, :, :]
	uy_real = real_images[:, 1, :, :]


	# load_dir_path = "/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/one_seventh/rbc_3500/noramlized_by_max/"
	load_path =  "/data0/rgupta2/dataset/rbc_500/max_normalized/tke_average_energies_{}.npy".format(res)


	my_dict_back = np.load(load_path, allow_pickle=True)	


	ux_average_over_time = my_dict_back.item()["256_ux"]
	uy_average_over_time = my_dict_back.item()["256_uy"]
	# max_value = my_dict_back.item()["max_value"]
 


	tke_gen = ((ux_data[0] - ux_average_over_time)**2 + (uy_data[0] - uy_average_over_time)**2)
	tke_real = ((ux_real[0] - ux_average_over_time)**2 + (uy_real[0] - uy_average_over_time)**2)

	sp1D_gen = tke2spectrum(tke_gen, res)
	sp1D_real = tke2spectrum(tke_real, res)

	return sp1D_gen, sp1D_real
	# plot with various axes scales


"""parsing and configuration"""
def parse_args():
	desc = "Tensorflow implementation of StyleGAN"
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--phase', type=str, default='test', help='[train, test, draw]')
	parser.add_argument('--draw', type=str, default='uncurated', help='[uncurated, style_mix, truncation_trick]')
	parser.add_argument('--dataset', type=str, default= "rbc_500", help='The dataset name what you want to generate')

	parser.add_argument('--iteration', type=int, default=120, help='The number of images used in the train phase')
	parser.add_argument('--max_iteration', type=int, default=2500, help='The total number of images')

	parser.add_argument('--batch_size', type=int, default=1, help='The size of batch in the test phase')
	parser.add_argument('--gpu_num', type=int, default=1, help='The number of gpu')

	parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
	parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')

	parser.add_argument('--start_res', type=int, default=8, help='The number of starting resolution')
	parser.add_argument('--img_size', type=int, default=256, help='The target size of image')
	parser.add_argument('--test_num', type=int, default=50, help='The number of generating images in the test phase')
	parser.add_argument('--input_channels', type=int, default=2, help='The number of input channels for the input real images')
	parser.add_argument('--seed', type=str2bool, default=True, help='seed in the draw phase')

	
	parser.add_argument('--checkpoint_dir', type=str, default='/data0/rgupta2/wo_style_rbc_model_weights/',
						help='Directory name to save the checkpoints')
	
	parser.add_argument('--result_dir', type=str, default='/data0/rgupta2/wo_style_rbc_model_weights/results/',help='Directory name to save the generated images')
	
	parser.add_argument('--log_dir', type=str, default='/data0/rgupta2/wo_style_rbc_model_weights/logs',
						help='Directory name to save training logs')


	parser.add_argument('--sample_dir', type=str, default='/data0/rgupta2/wo_style_rbc_model_weights/samples',help='Directory name to save the samples on training')


	parser.add_argument('--style_mixing_flag', type=bool, default= False,
						help='should there be style mixing of two latents from g_mapping network , default is false')

	"""
	/global/cscratch1/sd/rgupta2/backup/StyleGAN/src/stored_outputs/one_seventh_divergence/result/single_generator_results
	"""


	# parser.add_argument('--saved_numpy_file_of_rbc_images', type=str, default="single_generator_results/generator_386093_images_50_file_1.npy" , help="numpy file from which the latents are sourced")

	parser.add_argument('--dataset_location', type=str, default="/data0/rgupta2/dataset/rbc_500/max_normalized/", help='dataset_directory')

	parser.add_argument('--name_experiment', type=str, default="", help='generating noise levels at different resolutions')


	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

	# import comet_ml in the top of your file



	experiment = Experiment(api_key="YC7c0hMcGsJyRRjD98waGBcVa",
								project_name="post_analyzing_{}".format(args.name_experiment, args.dataset), workspace="style-gan")



	hyper_params = vars(args)
	experiment.log_parameters(hyper_params)
	# --checkpoint_dir
	check_folder(args.checkpoint_dir)

	# --result_dir
	check_folder(args.result_dir)

	# --result_dir
	check_folder(args.log_dir)

	# --sample_dir
	check_folder(args.sample_dir)

	# --batch_size
	try:
		assert args.batch_size >= 1
	except:
		print('batch size must be larger than or equal to one')
	return args, experiment


"""main"""
def main():
	# parse arguments
	args, experiment = parse_args()
	
	experiment.set_name(" running noise level experiments ")
	if args is None:
		exit()

	# open session
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:


	generated_data_location = sorted(glob("/data0/rgupta2/wo_style_rbc_model_weights/results/generator_354218_images_50_file_*.npy"))

	real_data_location = sorted(glob("/data0/rgupta2/dataset/rbc_500/max_normalized/*.npy"))[1]
	# [-1]

	# generated_data = np.load(generated_data_location, allow_pickle=True).item()["generated_images"]




	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth = True)
	
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
		with experiment.train():
			gan = StyleGAN(sess, args, experiment)

			# build graph
			gan.build_model()

			# show network architecture
			show_all_variables()
			gan.draw_style_mixing_figure(generated_data_location, real_data_location)
			print(" [*] Test finished!")




if __name__ == '__main__':
	main()





 



