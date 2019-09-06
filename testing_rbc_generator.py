from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
	
from comet_ml import Experiment

# from StyleGAN import StyleGAN

import argparse
from utils import *



# Add the following code anywhere in your machine learning file
# experiment = Experiment(api_key="YC7c0hMcGsJyRRjD98waGBcVa",
# 						project_name="on-celeba", workspace="style-gan")

import time, re, sys
from ops import *
from utils import *
import tensorflow
print(tensorflow.__version__)
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import PIL.Image
from tqdm import tqdm

class StyleGAN(object):

	def __init__(self, sess, args, experiment):

		self.experiment = experiment
		self.phase = args.phase
		self.progressive = args.progressive
		self.model_name = "StyleGAN"
		self.sess = sess
		self.dataset_name = args.dataset
		self.checkpoint_dir = args.checkpoint_dir
		self.sample_dir = args.sample_dir
		self.result_dir = args.result_dir
		self.log_dir = args.log_dir

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

		if(self.rbc_data):
			self.dataset = load_from_numpy(self.dataset_name, self.dataset_location)
			self.dataset_num = 3500 * len(self.dataset)
		else:
			self.dataset = load_data(dataset_name=self.dataset_name)
			self.dataset_num = len(self.dataset)

		# self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
		check_folder(self.sample_dir)

		print("##### Information #####")
		print("# dataset : ", self.dataset_name)
		print("# dataset number : ", self.dataset_num)
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


				# if(self.rbc_data):
				# 	images_out = x
				# else:
				
				

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
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)
					
					# if(self.rbc_data):
					# 	img = x
					# else:
					img = torgb(x, res, self.input_channels, sn=self.sn)
					
					images_out = upscale2d(images_out)
					images_out = smooth_transition(images_out, img, res, resolutions[-1], alpha)

					layer_index += 2

			else :
				for res, n_f in zip(resolutions[1:], featuremaps[1:]) :
					x = synthesis_block(x, res, w_broadcasted, layer_index, n_f, sn=self.sn)

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

		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
		print(" [*] Reading checkpoints from  {} with counter {} ".format(checkpoint_dir, counter))

		if(counter == 0):

			states = tf.train.get_checkpoint_state(checkpoint_dir)
			checkpoint_paths = states.all_model_checkpoint_paths
			load_model_path = checkpoint_paths[-1]

		else:
			load_model_path =  os.path.join(checkpoint_dir,  self.model_name) + "/StyleGAN.model-{}".format(counter)




		# "../stored_outputs/one_seventh_divergence/checkpoint/StyleGAN_rbc_3500_8to256_progressive/StyleGAN.model-315468"

		try:
			self.saver.restore(self.sess, load_model_path)
			print(" [*] Success to read {}".format(load_model_path))
			counter = int(load_model_path.split('-')[-1])
			return True, counter

		except:

			print(" [*] Failed to find a checkpoint")
			return False, -1

		# # ckpt = tf.train.checkpoint(checkpoint_dir, latest_filename= "StyleGAN.model-"+ str(counter) + ".data-00000-of-00001")

		# # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

		# if ckpt and ckpt.model_checkpoint_path:

		# 	ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
		# 	self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
		# 	counter = int(ckpt_name.split('-')[-1])
		
	

	def test(self, counter):

		print("Entered test phase successfully")
		tf.global_variables_initializer().run()

		print("intitialized tensorflow variable")

		self.saver = tf.train.Saver()
		
		# could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		
		could_load, checkpoint_counter = self.load(self.checkpoint_dir, counter)

		result_dir = os.path.join(self.result_dir, "single_generator_results/")
		check_folder(result_dir)

		if could_load:
			print(" [*] Load SUCCESS with checkpoint counter {}".format(checkpoint_counter))
		else:
			print(" [!] Load failed...")

		image_frame_dim = int(np.floor(np.sqrt(self.batch_size)))





		# tf_dict_multi_gpu = multi_gpu(num_gpus)
		# tf.global_variables_initializer().run()
		# batches = create_input_batches(num_gpus)
		# multi_gpu_inputs = create_multi_gpu_batches(batches, num_gpus)
		
		# for input_batch in multi_gpu_inputs:
		# 	input_feed_dict = {}
		# 	output_multi_gpus = []
		# 	for idx in range(len(tf_dict_multi_gpu)):
		# 		output_multi_gpus.append(tf_dict_multi_gpu[idx]['out'])












		generated_image = []
		saved_seeds = []

		Data = {}
		for i in tqdm(range(self.test_num)):

			# if self.batch_size == 1:


			seed = np.random.randint(low=0, high=10000)
			test_z = tf.cast(np.random.RandomState(seed).normal(size=[self.batch_size, self.z_dim]), tf.float32)
			alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
			self.fake_images = self.generator(test_z, alpha=alpha, target_img_size=self.img_size, is_training=False)
			samples = self.sess.run(self.fake_images)
			generated_image.append(samples)
			saved_seeds.append(seed)

		generated_images = np.concatenate( generated_image, axis=0 )
		seeds = np.asarray(saved_seeds)
		
		Data["generated_images"] = generated_images
		Data["seeds"] = seeds

				
		# if not os.path.exists(result_dir):
		# 	os.makedirs(result_dir)
		
		trial = 0
		save_file_path = str(result_dir) + "generator_{}_images_{}_file_{}.npy".format(checkpoint_counter, self.test_num, trial)
		while(os.path.isfile(save_file_path)):
			trial += 1
			save_file_path = str(result_dir) + "generator_{}_images_{}_file_{}.npy".format(checkpoint_counter, self.test_num, trial )

		np.save( save_file_path, Data)
		print("Operation completed")

"""

load_path = os.path.join(dataset_location + "tke_average_energies.npy")
my_dict_back = np.load(load_path, allow_pickle=True)



ux_average_over_time = my_dict_back.item()["{}_ux".format(image_size)]
uy_average_over_time = my_dict_back.item()["{}_uy".format(image_size)]



"""


# def multi_gpu_model_parallelism(gan, num_gpus=2):
# 	tf_dict = []
# 	for i in range(num_gpus):
# 		with tf.device('/gpu:{}'.format(i)):
# 			tf_dict.append(gan.build_graph)





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
	parser.add_argument('--gpu_num', type=int, default=2, help='The number of gpu')

	parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
	parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')

	parser.add_argument('--start_res', type=int, default=8, help='The number of starting resolution')
	parser.add_argument('--img_size', type=int, default=256, help='The target size of image')
	
	parser.add_argument('--test_num', type=int, default=50, help='The number of generating images in the test phase')
	
	parser.add_argument('--input_channels', type=int, default=2, help='The number of input channels for the input real images')
	parser.add_argument('--seed', type=str2bool, default=True, help='seed in the draw phase')

	parser.add_argument('--checkpoint_dir', type=str, default='../stored_outputs/checkpoint',
						help='Directory name to save the checkpoints')
	parser.add_argument('--result_dir', type=str, default='../stored_outputs/results',
						help='Directory name to save the generated images')
	parser.add_argument('--log_dir', type=str, default='../stored_outputs/logs',
						help='Directory name to save training logs')
	parser.add_argument('--sample_dir', type=str, default='../stored_outputs/samples',
						help='Directory name to save the samples on training')


	parser.add_argument('--style_mixing_flag', type=bool, default= False,
						help='should there be style mixing of two latents from g_mapping network , default is false')



	parser.add_argument('--dataset_location', type=str, default="/global/cscratch1/sd/rgupta2/backup/StyleGAN/dataset/rbc_500/max/",
						help='dataset_directory')


	parser.add_argument('--name_experiment', type=str, default="",
						help='dataset_directory')


	parser.add_argument('--counter_number', type=int, default=0,
						help='number of the model to be loaded (0 makes it load the latest model)')



	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

	# import comet_ml in the top of your file

	experiment = Experiment(api_key="YC7c0hMcGsJyRRjD98waGBcVa",
								project_name="inference_{}".format(args.name_experiment, args.dataset), workspace="style-gan")



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
	
	experiment.set_name(" generating rbc images ")
	if args is None:
		exit()

	# open session
	# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth = True)
	
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
		with experiment.train():
			gan = StyleGAN(sess, args, experiment)

			# build graph
			gan.build_model()

			# show network architecture
			show_all_variables()
			gan.test(args.counter_number)
			print(" [*] Test finished!")

			# experiment.set_model_graph(sess.graph)

			# if args.phase == 'train' :
			# 	# launch the graph in a session
			# 	gan.train()
			# 	print(" [*] Training finished!")

		
			# if args.phase == 'draw' :
			# 	if args.draw == 'style_mix' :
			# 		gan.draw_style_mixing_figure()
			# 		print(" [*] Style mix finished!")

			# 	elif args.draw == 'truncation_trick' :
			# 		gan.draw_truncation_trick_figure()
			# 		print(" [*] Truncation_trick finished!")

			# 	else :
			# 		gan.draw_uncurated_result_figure()
			# 		print(" [*] Un-curated finished!")




"""

 python  /global/cscratch1/sd/rgupta2/backup/StyleGAN/src/StyleGAN-Tensorflow/testing_rbc_generator.py --dataset rbc_500 --input_channels 2 --start_res 8 \
	--img_size 256 --gpu_num 8 --progressive True --phase train \
	--checkpoint_dir ../stored_outputs/wo_style_rbc/checkpoint --result_dir ../stored_outputs/wo_style_rbc/result \
	--log_dir ../stored_outputs/wo_style_rbc/log --sample_dir ../stored_outputs/wo_style_rbc/sample  --name_experiment "without_style_mixing_max_norm"

"""


if __name__ == '__main__':
	main()





 


