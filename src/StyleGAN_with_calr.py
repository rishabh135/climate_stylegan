import time, re, sys
from ops import *
from utils import *
import tensorflow



"""

wgan with gradient penalty

keep the general paradigm of non-saturating loss, with add wasserstein loss : 

check the shapes of final layers of discriminator 
the output of w-gan would be twice due to gradient penalty

"""

print(tensorflow.__version__)

from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import PIL.Image
from tqdm import tqdm

from RAdam import RAdamOptimizer


import clr

class StyleGAN(object):

	def __init__(self, sess, args, experiment):


		self.experiment = experiment
		self.phase = args.phase
		self.progressive = args.progressive

		self.model_name = "Climate-	StyleGAN"
		self.sess = sess
		self.dataset_name = args.dataset
		self.checkpoint_dir = args.checkpoint_dir
		self.sample_dir = args.sample_dir
		self.result_dir = args.result_dir	
		self.log_dir = args.log_dir

		self.iteration = args.iteration * 10000
		self.max_iteration = args.max_iteration * 10000

		self.climate_img_size = args.climate_img_size
		self.batch_size = args.batch_size
		self.img_size = args.img_size

		""" Hyper-parameter"""
		self.start_res = args.start_res
		self.resolutions = resolution_list(self.img_size) # [4, 8, 16, 32, 64, 128, 256, 512, 1024 ...]
		self.featuremaps = featuremap_list(self.img_size) # [512, 512, 512, 512, 256, 128, 64, 32, 16 ...]

		if not self.progressive :
			self.resolutions = [self.resolutions[-1]]
			self.featuremaps = [self.featuremaps[-1]]
			self.start_res = self.resolutions[-1]

		self.gpu_num = args.gpu_num



		"""
		parameters to experiment with

		"""

		"""
		removing validation phase from the model, i.e. no more saving files at differnet resolutions a


		"""

		self.store_images_flag = False
		self.style_mixing_flag = args.style_mixing_flag
		self.divergence_lambda = 0.00
		self.divergence_loss_flag = False
		self.plotting_histogram_images = 64


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
		self.batch_sizes = get_batch_sizes(self.gpu_num)

		"""
		if gpu_num == 4:
		x = OrderedDict([(4, 128), (8, 64), (16, 32), (32, 16), (64, 8), (128, 4), (256, 4), (512, 4), (1024, 4)])

		"""
		self.end_iteration = get_end_iteration(self.iteration, self.max_iteration, self.train_with_trans, self.resolutions, self.start_res)
		


		self.g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
		self.d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

		self.sn = args.sn

		self.print_freq = {4: 1000, 8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 3000, 256: 5000, 512: 10000, 1024: 10000}

		# self.print_freq = {4: 1000, 8: 500, 16: 500, 32: 500, 64: 100, 128: 100, 256: 10, 512: 10000, 1024: 10000}

		self.save_freq = {4: 1000, 8: 1000, 16: 1000, 32: 1000, 64: 1000, 128: 3000, 256: 5000, 512: 10000, 1024: 10000}

		self.print_freq.update((x, y // self.gpu_num) for x, y in self.print_freq.items())
		self.save_freq.update((x, y // self.gpu_num) for x, y in self.save_freq.items())

		self.test_num = args.test_num
		self.seed = args.seed



		self.climate_data = self.dataset_name.startswith("climate")
		self.dataset_location = args.dataset_location

		if(self.climate_data):
			self.dataset = load_from_numpy(self.dataset_name, self.dataset_location)

			"""
			dataset_num is approximate based on number of samples for one full year, note that all files do not correspond to a single year

			"""
			self.dataset_num = 2920 * len(self.dataset)
		else:
			self.dataset = load_data(dataset_name=self.dataset_name)
			self.dataset_num = len(self.dataset)




		self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
		check_folder(self.sample_dir)

		print("\n\n")

		print("##### Information #####")
		print("# dataset : ", self.dataset_name)
		# print("# dataset number : ", self.dataset_num)
		print("# gpu : ", self.gpu_num)
		print("# batch_size in train phase : ", self.batch_sizes)
		print("# batch_size in test phase : ", self.batch_size)
		print( "self.end_iteration : ", self.end_iteration)
		print("resolutions: ", self.resolutions)
		print("feature maps: ", self.featuremaps)

		print("# start resolution : ", self.start_res)
		print("# target resolution : ", self.img_size)
		print("# iteration per resolution : ", self.iteration)

		print("# progressive training : ", self.progressive)
		print("# spectral normalization : ", self.sn)
		print("# store images: {}".format(self.store_images_flag))
		print("# use divergence in loss: {}".format(self.divergence_loss_flag))

		print("\n\n")

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

	def g_synthesis(self, w_broadcasted, alpha, resolutions, featuremaps, reuse=tf.AUTO_REUSE):
		with tf.variable_scope('g_synthesis', reuse=reuse):
			coarse_styles, middle_styles, fine_styles = get_style_class(resolutions, featuremaps)
			layer_index = 2

			""" initial layer """
			res = resolutions[0]
			n_f = featuremaps[0]

			x = synthesis_const_block(res, w_broadcasted, n_f, self.sn)

			""" remaining layers """
			if self.progressive :

				images_out = torgb(x, res, input_channels=self.input_channels, sn=self.sn)
				coarse_styles.pop(res, None)

				# Coarse style [4 ~ 8]
				# pose, hair, face shape
				for res, n_f in coarse_styles.items():


					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)


					img = torgb(x, res, self.input_channels, sn=self.sn)

				
					images_out = upscale2d(images_out)

					
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

					

					layer_index += 2


				# print("########################\n\n")
				# Middle style [16 ~ 32]
				# facial features, eye
				for res, n_f in middle_styles.items():
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)

					img = torgb(x, res, self.input_channels,  sn=self.sn)
					

					images_out = upscale2d(images_out)
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

					layer_index += 2

				# Fine style [64 ~ 1024]
				# color scheme
				for res, n_f in fine_styles.items():
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)
					

					img = torgb(x, res, self.input_channels, sn=self.sn)					
					images_out = upscale2d(images_out)
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

					layer_index += 2

			else :
				for res, n_f in zip(resolutions[1:], featuremaps[1:]) :
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)

					layer_index += 2

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

			x = fromrgb(x_init, r_resolutions[0], r_featuremaps[0], self.sn)


			""" stack discriminator blocks """
			for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
				res_next = r_resolutions[index + 1]
				n_f_next = r_featuremaps[index + 1]

				x = discriminator_block(x, res, n_f, n_f_next, self.sn)

				if self.progressive :
					x_init = downscale2d(x_init)					
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

		# print("Inside style mixing the w_broadcasted = {} and layer_indices {} ".format(w_broadcasted.shape, layer_indices))
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
		""" Graph """
		if self.phase == 'train' :
			self.d_loss_per_res = {}
			self.g_loss_per_res = {}

			self.r1_penalty = {}

			self.generator_optim = {}
			self.discriminator_optim = {}


			self.alpha_summary_per_res = {}
			self.d_summary_per_res = {}
			self.g_summary_per_res = {}

			self.alpha_stored_per_res = {}
			

			if(self.store_images_flag):
				self.fake_images = {}
				self.real_images = {}


			self.divergence_fake = {}
			self.divergence_real = {}


			for res in self.resolutions[self.resolutions.index(self.start_res):]:
				g_loss_per_gpu = []
				d_loss_per_gpu = []

				r1_penalty_list = []

				fake_images_per_gpu = []
				real_images_per_gpu = []
				


				batch_size = self.batch_sizes.get(res, self.batch_size_base)
				global_step = tf.get_variable('global_step_{}'.format(res), shape=[], dtype=tf.float32,
											  initializer=tf.initializers.zeros(),
											  trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)
				alpha_const, zero_constant = get_alpha_const(self.iteration // 2, batch_size * self.gpu_num, global_step)

				# smooth transition variable
				do_train_trans = self.train_with_trans[res]

				alpha = tf.get_variable('alpha_{}'.format(res), shape=[], dtype=tf.float32,
										initializer=tf.initializers.ones() if do_train_trans else tf.initializers.zeros(),
										trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

				if do_train_trans:
					alpha_assign_op = tf.assign(alpha, alpha_const)
				else:
					alpha_assign_op = tf.assign(alpha, zero_constant)


				with tf.control_dependencies([alpha_assign_op]):
					for gpu_id in range(self.gpu_num):
						with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
							with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
								# images
								gpu_device = '/gpu:{}'.format(gpu_id)
								image_class = ImageData(res)


								if(self.climate_data):
									"""
									npy_file is just to calculate the common header across all the files, in self.dataset 
									self.dataset = list of chosen files
									num_features = constant , the dimension of each sample from the index file
									"""
									
									npy_file = self.dataset[0]
									num_features = self.input_channels * self.climate_img_size * self.climate_img_size
									dtype = tf.float32
									header_offset = npy_header_offset(npy_file)
									dataset = tf.data.FixedLengthRecordDataset(self.dataset, num_features * dtype.size, header_bytes=header_offset)
									dataset = dataset.map(lambda s: tf.image.resize( tf.transpose(tf.reshape(tf.decode_raw(s, dtype), [self.input_channels,self.climate_img_size, self.climate_img_size]) , perm=[1,2,0]), size=[res, res], method=tf.image.ResizeMethod.BILINEAR), num_parallel_calls=4)
 




									# # rbc_class = RBCData(res, self.input_channels)
									# train_dataset = tf.data.Dataset.from_tensor_slices(self.dataset)	
									


									# inputs = train_dataset.apply(map_and_batch(rbc_class.image_processing, batch_size, num_parallel_batches=16, drop_remainder=True)). \
									# 	apply(prefetch_to_device(gpu_device, None))
									


									# inputs = train_dataset.map(lambda item: tf.py_func(read_npy_file, [item, res], [tf.float32, ]))
						
									inputs = dataset.apply(shuffle_and_repeat(self.dataset_num)).batch(batch_size, drop_remainder=True).apply(prefetch_to_device(gpu_device, None))

									# prefetch(buffer_size= tf.data.experimental.AUTOTUNE)
									
									inputs_iterator = inputs.make_one_shot_iterator()
									real_img  = inputs_iterator.get_next()
									real_img  = tf.reshape( real_img ,(batch_size, res , res, self.input_channels))
									# print("real img: ",type(real_img))									
									# print("real_img.shape: ",real_img.shape)
									# real_img.set_shape([None, self.img_size, None, self.input_channels])
									
	

									# inputs = tf.data.Dataset.from_tensor_slices(self.dataset)
									# ######################################################################################
									# # applying operations to input slices 
									# inputs = inputs. \
									# 	apply(shuffle_and_repeat(self.dataset_num)). \
									# 	apply(map_and_batch(image_class.image_processing, batch_size, num_parallel_batches=16, drop_remainder=True)). \
									# 	apply(prefetch_to_device(gpu_device, None))
									# When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size

								else:


									inputs = tf.data.Dataset.from_tensor_slices(self.dataset)
									######################################################################################
									# applying operations to input slices 
									inputs = inputs.apply(shuffle_and_repeat(self.dataset_num)). \
										apply(map_and_batch(image_class.image_processing, batch_size, num_parallel_batches=8, drop_remainder=True)). \
										apply(prefetch_to_device(gpu_device, None))
									
									# When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size
									inputs_iterator = inputs.make_one_shot_iterator()
									real_img = inputs_iterator.get_next()
									# print("real img: ",type(real_img))									
									# print("real_img.shape: ",real_img.shape)


								z = tf.random_normal(shape=[batch_size, self.z_dim])

								

								fake_img = self.generator(z, alpha, res)
								# print("fake_img: ",fake_img.shape)
								

								# if(not self.climate_data):
								real_img = smooth_crossfade(real_img, alpha)


								# print("real_img: ",real_img.shape)

								real_logit = self.discriminator(real_img, alpha, res)




								# print("#########################\n\n\n\n")
								# print("starting fake logits now")
								fake_logit = self.discriminator(fake_img, alpha, res)
								# print("****** fake_logit shape {} ".format(fake_logit.shape))
								# compute loss
								d_loss, g_loss, r1_penalty = compute_loss(real_img, real_logit, fake_logit)

								d_loss_per_gpu.append(d_loss)
								g_loss_per_gpu.append(g_loss)
								# print(" g_loss : {} and r1 pentalty : {} ".format(g_loss.shape, r1_penalty.shape))


								# print("\n\n\n")
								r1_penalty_list.append(r1_penalty)
								
								fake_images_per_gpu.append(fake_img)
								real_images_per_gpu.append(real_img)

				# print("Create graph for {} resolution".format(res))


				if(self.divergence_loss_flag):
					self.divergence_fake[res], self.divergence_real[res] = calculate_divergence_tf( tf.concat(fake_images_per_gpu, axis=0)[-batch_size * self.gpu_num:], tf.concat(real_images_per_gpu, axis=0)[-batch_size * self.gpu_num:])

				# prepare appropriate training vars
				d_vars, g_vars = filter_trainable_variables(res)

				d_loss = tf.reduce_mean(d_loss_per_gpu)
				g_loss = tf.reduce_mean(g_loss_per_gpu)

				"""
				adding divergence as an added loss to the generator

				"""

				if(self.divergence_loss_flag):
					g_loss += self.divergence_lambda * self.divergence_fake[res]

				self.r1_penalty[res] = tf.reduce_mean(r1_penalty_list)

				d_lr = self.d_learning_rates.get(res, self.learning_rate_base)
				g_lr = self.g_learning_rates.get(res, self.learning_rate_base)

				if self.gpu_num == 1 :
					colocate_grad = False
				else :
					colocate_grad = True

				



				# d_optim = RAdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, weight_decay=0.0)\
				# 			.minimize(d_loss,var_list=d_vars,colocate_gradients_with_ops=colocate_grad)



				d_optim = tf.train.AdamOptimizer( learning_rate=clr.cyclic_learning_rate(global_step=global_step, learning_rate = d_lr, mode='triangular2'), beta1=0.0, beta2=0.99, epsilon=1e-8).minimize(d_loss,var_list=d_vars,global_step=global_step,colocate_gradients_with_ops=colocate_grad)




				# d_optim = tf.train.AdamOptimizer(d_lr, beta1=0.0, beta2=0.99, epsilon=1e-8).minimize(d_loss,var_list=d_vars,colocate_gradients_with_ops=colocate_grad)

				# g_optim = RAdamOptimizer(learning_rate= 0.001, beta1=0.9, beta2=0.999, weight_decay=0.0)\
				# 	.minimize(g_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=colocate_grad)
				


				# g_optim = tf.train.AdamOptimizer(g_lr, beta1=0.0, beta2=0.99, epsilon=1e-8).minimize(g_loss,var_list=g_vars,global_step=global_step,colocate_gradients_with_ops=colocate_grad)

				"""
					trying cyclical annealed learning rate 

				"""

				g_optim = tf.train.AdamOptimizer( learning_rate=clr.cyclic_learning_rate(global_step=global_step, learning_rate = g_lr, mode='triangular2'), beta1=0.0, beta2=0.99, epsilon=1e-8).minimize(g_loss,var_list=g_vars,global_step=global_step,colocate_gradients_with_ops=colocate_grad)



				self.discriminator_optim[res] = d_optim
				self.generator_optim[res] = g_optim

				self.d_loss_per_res[res] = d_loss
				self.g_loss_per_res[res] = g_loss


				if(self.store_images_flag):
					self.fake_images[res] = tf.concat(fake_images_per_gpu, axis=0)
					self.real_images[res] = tf.concat(real_images_per_gpu, axis=0)









				self.alpha_stored_per_res[res] = alpha



				""" Summary """
				self.alpha_summary_per_res[res] = tf.summary.scalar("alpha_{}".format(res), alpha)

				self.d_summary_per_res[res] = tf.summary.scalar("d_loss_{}".format(res), self.d_loss_per_res[res])
				self.g_summary_per_res[res] = tf.summary.scalar("g_loss_{}".format(res), self.g_loss_per_res[res])

		else :
			"""" Testing """
			test_z = tf.random_normal(shape=[self.batch_size, self.z_dim])
			alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
			self.fake_images = self.generator(test_z, alpha=alpha, target_img_size=self.img_size, is_training=False)


	##################################################################################
	# Train
	##################################################################################

	def train(self):
		# initialize all variables
		tf.global_variables_initializer().run()

		# saver to save model
		self.saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)
		# self.experiment.set_model_graph(self.sess.graph)

		# summary writer
		self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)





		# restore check-point if it exits
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		
		print("checkpoint_counter: ", checkpoint_counter)

		if could_load:

			start_res_idx = get_checkpoint_res(checkpoint_counter, self.batch_sizes, self.iteration,
											   self.start_res, self.img_size, self.gpu_num,
											   self.end_iteration, self.train_with_trans)

			if not self.progressive :
				start_res_idx = 0

			start_batch_idx = checkpoint_counter

			# print("start_batch_idx inside could load :" , start_batch_idx)
			
			for res_idx in range(self.resolutions.index(self.start_res), start_res_idx) :


				res = self.resolutions[res_idx]
				batch_size_per_res = self.batch_sizes.get(res, self.batch_size_base) * self.gpu_num


				if self.train_with_trans[res]:
					if res == self.img_size :
						iteration = self.end_iteration
					else :
						iteration = self.iteration
				else :
					iteration = self.iteration // 2

				if start_batch_idx - (iteration // batch_size_per_res) < 0:
					break
				else:
					start_batch_idx = start_batch_idx - (iteration // batch_size_per_res)


			counter = checkpoint_counter
			print(" [*] Load SUCCESS")

		else:
			start_res_idx = self.resolutions.index(self.start_res)
			start_batch_idx = 0
			counter = 1
			print(" [!] Load failed...")

		start_time = time.time()





		for current_res_num in range(start_res_idx, len(self.resolutions)):



			self.generated_images_stored = []
			self.real_images_stored = []

			current_res = self.resolutions[current_res_num]
			batch_size_per_res = self.batch_sizes.get(current_res, self.batch_size_base) * self.gpu_num

			if self.progressive :
				if self.train_with_trans[current_res] :

					if current_res == self.img_size :
						current_iter = self.end_iteration // batch_size_per_res
					else :
						current_iter = self.iteration // batch_size_per_res
				else :
					current_iter = (self.iteration // 2) // batch_size_per_res

			else :
				current_iter = self.end_iteration

			for idx in range(start_batch_idx, current_iter):

				# update D network
				_, summary_d_per_res, d_loss = self.sess.run([self.discriminator_optim[current_res],
															  self.d_summary_per_res[current_res],
															  self.d_loss_per_res[current_res]])

				self.writer.add_summary(summary_d_per_res, idx)

				# update G network
				_, summary_g_per_res, summary_alpha, g_loss = self.sess.run([self.generator_optim[current_res],
																			 self.g_summary_per_res[current_res],
																			 self.alpha_summary_per_res[current_res],
																			 self.g_loss_per_res[current_res]])


				r1_loss, alpha = self.sess.run([self.r1_penalty[current_res],self.alpha_stored_per_res[current_res] ])

				# divergence_fake, divergence_real, alpha = self.sess.run([self.r1_penalty[current_res], self.divergence_fake[current_res], self.divergence_real[current_res], self.alpha_stored_per_res[current_res] ])

				self.writer.add_summary(summary_g_per_res, idx)
				self.writer.add_summary(summary_alpha, idx)

				# display training status
				self.experiment.set_step(counter)
				counter += 1

				self.experiment.log_metric("d_loss",d_loss,step=counter)
				self.experiment.log_metric("g_loss",g_loss,step=counter)
				self.experiment.log_metric("r1_penalty", r1_loss,step=counter)
				# self.experiment.log_metric("divergence_real", divergence_real, step=counter)
				# self.experiment.log_metric("divergence_fake", divergence_fake, step=counter)
				self.experiment.log_metric("alpha value ", alpha, step=counter)



				
				print("Current res: [%4d] [%6d/%6d] with current  time: %4.4f, d_loss: %.8f, g_loss: %.8f  alpha: %.4f  " \
					  % (current_res, idx, current_iter, time.time() - start_time, d_loss, g_loss, alpha))



				"""

				function to save iteration or not

				"""


				if np.mod(idx + 1, self.save_freq[current_res]) == 0:
					self.save(self.checkpoint_dir, counter)



				"""

				function to plot images and their divergence plots

				"""


				if ((np.mod(idx + 1, self.print_freq[current_res]) == 0) and self.store_images_flag):


					samples = self.sess.run(self.fake_images[current_res])

					"""
						Also plotting real samples for transparency purposes
					"""
					
					real_samples = self.sess.run(self.real_images[current_res])



					# fake_values, real_values = self.sess.run(self.divergence_fake[res], self.divergence_real[res]) 


					manifold_h = int(np.floor(np.sqrt(batch_size_per_res)))
					manifold_w = int(np.floor(np.sqrt(batch_size_per_res)))


					if(self.climate_data):

						"""
		
						### To DO a non hacky way to add subarrays continuously 

						"""
						# self.generated_images_stored.append(samples)
						# self.real_images_stored.append(real_samples)
						
						# try:
						# 	self.ux_images_stored.concatenate( axis=0)
						# except:
						# 	self.ux_images_stored = samples[:manifold_h * manifold_w, :, :, 0]
						

						# try:	
						# 	np.concatenate(self.uy_images_stored, samples[:manifold_h * manifold_w, :, :, 1], axis=0)
						# except:
						# 	self.uy_images_stored = samples[:manifold_h * manifold_w, :, :, 1]
							





						
						# self.experiment.log_metric("summary alpha", summary_alpha,step=counter)
						

						# try:
						# 	self.real_samples.concatenate(, axis=0)
						# except:
						# 	self.real_samples = real_samples


						# print("Current res: real images.shape : {} , generated shape : {}  ".format( self.real_images_stored.shape[0], self.generated_images_stored.shape[0] ))
							
						"""
						it now starts to check if the number of stored images is atleast > self.plotting_hitogram_images (which by default is 64)


						"""
						if(len(self.real_images_stored)* self.real_images_stored[0].shape[0] >= self.plotting_histogram_images):


							# print("^^^^^^^^^^^^^ it entered the plotting function correctly \n\n\n")
							tmp_real_images = np.concatenate(self.real_images_stored, axis=0)

							tmp_fake_images = np.concatenate(self.generated_images_stored, axis=0)


						
							
							plot_velocity_hist(tmp_fake_images[-self.plotting_histogram_images:, :,:,0], tmp_fake_images[-self.plotting_histogram_images:, :, :, 1] , "./{}/fake_img_histogram_{:04d}_{:06d}.jpg".format(self.sample_dir, current_res, idx + 1), self.experiment, tmp_real_images[-self.plotting_histogram_images:,:,:,:])


							plot_generated_velocity(tmp_fake_images[-self.plotting_histogram_images:, :,:,0], tmp_fake_images[-self.plotting_histogram_images:, :, :, 1] , "./{}/real_and_fake_img_{:04d}_{:06d}.jpg".format(self.sample_dir, current_res, idx + 1), self.experiment, tmp_real_images[-self.plotting_histogram_images:, :, :, :])


							if np.mod(idx + 1, 50 * self.print_freq[current_res]) == 0:

								images_to_be_saved = min(5000, tmp_fake_images.shape[0])

								np.save( "./{}/generated_images_{}_at_index_{}.npy".format(self.result_dir, current_res, idx), 								tmp_fake_images[-images_to_be_saved:, :,:,:])
								np.save( "./{}/real_images_{}_at_index_{}.npy".format(self.result_dir, current_res, idx), tmp_real_images[-images_to_be_saved:, :,:,:])

							# if(current_res == 256):
							# 	try:
							# 		plot_tke(tmp_fake_images[-self.plotting_histogram_images:, :,:,0], tmp_fake_images[-self.plotting_histogram_images:, :, :, 1] , "./{}/tke_sp1d_{:04d}_{:06d}.jpg".format(self.sample_dir, current_res, idx + 1), self.experiment, tmp_real_images[-self.plotting_histogram_images:, :, :, :])

							# 	except:
							# 		continue



				# if np.mod(idx + 1, 2*self.print_freq[current_res]) == 0:


				# 	fake_values, real_values = self.sess.run([self.divergence_fake[current_res], self.divergence_real[current_res]])

				# 	fig=plt.figure()
				# 	plt.hist([fake_values.flatten(), real_values.flatten() ], bins='fd', histtype='step', density=True, color=["green","red"])	

				# 	self.experiment.log_figure(figure=plt,  figure_name=" divergence plots at index {} res {} ".format(idx, current_res ))



							# calculate_divergence(tmp_fake_images[-self.plotting_histogram_images:, :, :, :] , "./{}/real_and_fake_img_{:04d}_{:06d}.jpg".format(self.sample_dir, current_res, idx + 1), self.experiment, tmp_real_images[-self.plotting_histogram_images:, :, :, :])
					
						# exp_figure = imsave(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],  './{}/fake_img_{:04d}_{:06d}.jpg'.format(self.sample_dir, current_res, idx + 1), self.climate_data, current_res, self.dataset_location, self.experiment, self.num_images_to_be_shown, real_samples[:, :, :, :] )
					
						# self.experiment.log_figure(figure=exp_figure,  figure_name="fake images at {} at index {}".format(current_res, idx+1))


					# else:		
					# 	self.experiment.log_image( (samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], current_res ), name="fake images generated during training")
					# 	save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
					# 				'./{}/fake_img_{:04d}_{:06d}.jpg'.format(self.sample_dir, current_res, idx + 1), self.climate_data)



			# After an epoch, start_batch_idx is set to zero
			# non-zero value is only for the first epoch after loading pre-trained model
			start_batch_idx = 0


			"""
				Save numpy files of original and generated images 
				shape :  NHWC    (c=2 in this case)
						
			"""
			# save model
			self.save(self.checkpoint_dir, counter)


		# save model for final step
		self.save(self.checkpoint_dir, counter)

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

	def save(self, checkpoint_dir, step):
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(ckpt_name.split('-')[-1])
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0

	def test(self):
		tf.global_variables_initializer().run()

		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		result_dir = os.path.join(self.result_dir, self.model_dir)
		check_folder(result_dir)

		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		image_frame_dim = int(np.floor(np.sqrt(self.batch_size)))

		for i in tqdm(range(self.test_num)):

			if self.batch_size == 1:
				seed = np.random.randint(low=0, high=10000)
				test_z = tf.cast(np.random.RandomState(seed).normal(size=[self.batch_size, self.z_dim]), tf.float32)
				alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
				self.fake_images = self.generator(test_z, alpha=alpha, target_img_size=self.img_size, is_training=False)
				samples = self.sess.run(self.fake_images)

				visualize_each_image_test( samples[:image_frame_dim * image_frame_dim, :, :, :], [ image_frame_dim,  image_frame_dim],  './{}/fake_img_{:04d}_{:06d}.jpg'.format(result_dir, self.img_size, i),  self.experiment, self.dataset_location,  num_images_to_be_shown=4)	

			else :
				samples = self.sess.run(self.fake_images)


				if(self.climate_data):



					visualize_each_image_test( samples[:image_frame_dim * image_frame_dim, :, :, :], [ image_frame_dim,  image_frame_dim],  './{}/fake_img_{:04d}_{:06d}.jpg'.format(result_dir, self.img_size, i),  self.experiment, self.dataset_location,  num_images_to_be_shown=4)				
					

					# else:		
					# 	self.experiment.log_image( (samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w], current_res ), name="fake images generated during training")
					# 	save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
					# 				'./{}/fake_img_{:04d}_{:06d}.jpg'.format(self.sample_dir, current_res, idx + 1), self.climate_data)

				# save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
				# 			'{}/test_fake_img_{}_{}.jpg'.format(result_dir, self.img_size, i))

	def draw_uncurated_result_figure(self):

		tf.global_variables_initializer().run()
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		result_dir = os.path.join(self.result_dir, self.model_dir, 'paper_figure')
		check_folder(result_dir)

		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		lods = [0, 1, 2, 2, 3, 3]
		seed = 3291
		rows = 3
		cx = 0
		cy = 0

		alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

		if self.seed :
			latents = tf.cast(np.random.RandomState(seed).normal(size=[sum(rows * 2 ** lod for lod in lods), self.z_dim]), tf.float32)
		else :
			latents = tf.cast(np.random.normal(size=[sum(rows * 2 ** lod for lod in lods), self.z_dim]), tf.float32)

		images = self.sess.run(self.generator(latents, alpha=alpha, target_img_size=self.img_size, is_training=False))

		for i in range(len(images)) :
			images[i] = post_process_generator_output(images[i])


		canvas = PIL.Image.new('RGB', (sum(self.img_size // 2 ** lod for lod in lods), self.img_size * rows), 'white')
		image_iter = iter(list(images))

		for col, lod in enumerate(lods):
			for row in range(rows * 2 ** lod):
				image = PIL.Image.fromarray(np.uint8(next(image_iter)), 'RGB')

				image = image.crop((cx, cy, cx + self.img_size, cy + self.img_size))
				image = image.resize((self.img_size // 2 ** lod, self.img_size // 2 ** lod), PIL.Image.ANTIALIAS)
				canvas.paste(image, (sum(self.img_size // 2 ** lod for lod in lods[:col]), row * self.img_size // 2 ** lod))

		canvas.save('{}/figure02-uncurated.jpg'.format(result_dir))

	def draw_style_mixing_figure(self):
		tf.global_variables_initializer().run()
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		result_dir = os.path.join(self.result_dir, self.model_dir, 'paper_figure')
		check_folder(result_dir)

		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		src_seeds = [604, 8440, 7613, 6978, 3004]
		dst_seeds = [1336, 6968, 607, 728, 7036, 9010]

		resolutions = resolution_list(self.img_size)
		featuremaps = featuremap_list(self.img_size)
		n_broadcast = len(resolutions) * 2

		style_ranges = [range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, n_broadcast)]

		alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
		if self.seed :
			src_latents = tf.cast(np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.z_dim]) for seed in src_seeds), axis=0), tf.float32)
			dst_latents = tf.cast(np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.z_dim]) for seed in dst_seeds), axis=0), tf.float32)

		else :
			src_latents = tf.cast(np.random.normal(size=[len(src_seeds), self.z_dim]), tf.float32)
			dst_latents = tf.cast(np.random.normal(size=[len(dst_seeds), self.z_dim]), tf.float32)

		with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) :

			src_dlatents = self.g_mapping(src_latents, n_broadcast)
			dst_dlatents = self.g_mapping(dst_latents, n_broadcast)

			dlatent_avg = tf.get_variable('w_avg')

			src_dlatents = self.truncation_trick(n_broadcast, src_dlatents, dlatent_avg, self.truncation_psi)
			dst_dlatents = self.truncation_trick(n_broadcast, dst_dlatents, dlatent_avg, self.truncation_psi)

			src_images = self.sess.run(self.g_synthesis(src_dlatents, alpha, resolutions, featuremaps))
			dst_images = self.sess.run(self.g_synthesis(dst_dlatents, alpha, resolutions, featuremaps))

			for i in range(len(src_images)):
				src_images[i] = post_process_generator_output(src_images[i])

			for i in range(len(dst_images)):
				dst_images[i] = post_process_generator_output(dst_images[i])

			src_dlatents = self.sess.run(src_dlatents)
			dst_dlatents = self.sess.run(dst_dlatents)

			canvas = PIL.Image.new('RGB', (self.img_size * (len(src_seeds) + 1), self.img_size * (len(dst_seeds) + 1)), 'white')

			for col, src_image in enumerate(list(src_images)):
				canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * self.img_size, 0))

			for row, dst_image in enumerate(list(dst_images)):
				canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * self.img_size))

				row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
				row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]

				row_images = self.sess.run(self.g_synthesis(tf.convert_to_tensor(row_dlatents, tf.float32), alpha, resolutions, featuremaps))

				for i in range(len(row_images)):
					row_images[i] = post_process_generator_output(row_images[i])

				for col, image in enumerate(list(row_images)):
					canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'), ((col + 1) * self.img_size, (row + 1) * self.img_size))

			canvas.save('{}/figure03-style-mixing.jpg'.format(result_dir))

	def draw_truncation_trick_figure(self):

		tf.global_variables_initializer().run()
		self.saver = tf.train.Saver()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		result_dir = os.path.join(self.result_dir, self.model_dir, 'paper_figure')
		check_folder(result_dir)

		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		seeds = [1653, 4010]
		psis = [1, 0.7, 0.5, 0, -0.5, -1]

		resolutions = resolution_list(self.img_size)
		featuremaps = featuremap_list(self.img_size)
		n_broadcast = len(resolutions) * 2

		alpha = tf.constant(0.0, dtype=tf.float32, shape=[])

		if self.seed :
			latents = tf.cast(np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.z_dim]) for seed in seeds), axis=0), tf.float32)
		else :
			latents = tf.cast(np.random.normal(size=[len(seeds), self.z_dim]), tf.float32)

		with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

			dlatents = self.sess.run(self.g_mapping(latents, n_broadcast))
			dlatent_avg = tf.get_variable('w_avg')

			canvas = PIL.Image.new('RGB', (self.img_size * len(psis), self.img_size * len(seeds)), 'white')

			for row, dlatent in enumerate(list(dlatents)):

				row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
				row_images = self.sess.run(self.g_synthesis(row_dlatents, alpha, resolutions, featuremaps))

				for i in range(len(row_images)):
					row_images[i] = post_process_generator_output(row_images[i])

				for col, image in enumerate(list(row_images)):
					canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'), (col * self.img_size, row * self.img_size))

			canvas.save('{}/figure08-truncation-trick.jpg'.format(result_dir))
