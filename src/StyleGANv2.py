
import time, re, sys
from ops import *
from utils import *
import tensorflow
tensorflow.get_logger().setLevel('INFO')

from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import PIL.Image
from tqdm import tqdm
from data_pipeline import build_input_pipeline
from power_spectra import batch_power_spectrum

















from PIL import Image
from math import floor, log2
import numpy as np
import time
from functools import partial
from random import random
import os

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import tensorflow.keras.backend as K

from datagen import dataGenerator, printProgressBar
from conv_mod import *

im_size = 256
latent_size = 512
BATCH_SIZE = 16
directory = "Earth"

cha = 24

n_layers = int(log2(im_size) - 1)

mixed_prob = 0.9

def noise(n):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n):
    return [noise(n)] * n_layers

def mixedList(n):
    tt = int(random() * n_layers)
    p1 = [noise(n)] * tt
    p2 = [noise(n)] * (n_layers - tt)
    return p1 + [] + p2

def nImage(n):
    return np.random.uniform(0.0, 1.0, size = [n, im_size, im_size, 1]).astype('float32')


#Loss functions
def gradient_penalty(samples, output, weight):
    gradients = K.gradients(output, samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    # (weight / 2) * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty) * weight

def hinge_d(y_true, y_pred):
    return K.mean(K.relu(1.0 + (y_true * y_pred)))

def w_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


#Lambdas
def crop_to_fit(x):

    height = x[1].shape[1]
    width = x[1].shape[2]

    return x[0][:, :height, :width, :]

def upsample(x):
    return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')

def upsample_to_size(x):
    y = im_size / x.shape[2]
    x = K.resize_images(x, y, y, "channels_last",interpolation='bilinear')
    return x


#Blocks
def g_block(inp, istyle, inoise, fil, u = True):

    if u:
        #Custom upsampling because of clone_model issue
        out = Lambda(upsample, output_shape=[None, inp.shape[2] * 2, inp.shape[2] * 2, None])(inp)
    else:
        out = Activation('linear')(inp)

    rgb_style = Dense(fil, kernel_initializer = VarianceScaling(200/out.shape[2]))(istyle)
    style = Dense(inp.shape[-1], kernel_initializer = 'he_uniform')(istyle)
    delta = Lambda(crop_to_fit)([inoise, out])
    d = Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2DMod(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([out, style])
    out = add([out, d])
    out = LeakyReLU(0.2)(out)

    style = Dense(fil, kernel_initializer = 'he_uniform')(istyle)
    d = Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2DMod(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([out, style])
    out = add([out, d])
    out = LeakyReLU(0.2)(out)

    return out, to_rgb(out, rgb_style)

def d_block(inp, fil, p = True):

    res = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)

    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    out = LeakyReLU(0.2)(out)
    out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
    out = LeakyReLU(0.2)(out)

    out = add([res, out])

    if p:
        out = AveragePooling2D()(out)

    return out

def to_rgb(inp, style):
    size = inp.shape[2]
    x = Conv2DMod(3, 1, kernel_initializer = VarianceScaling(200/size), demod = False)([inp, style])
    return Lambda(upsample_to_size, output_shape=[None, im_size, im_size, None])(x)

def from_rgb(inp, conc = None):
    fil = int(im_size * 4 / inp.shape[2])
    z = AveragePooling2D()(inp)
    x = Conv2D(fil, 1, kernel_initializer = 'he_uniform')(z)
    if conc is not None:
        x = concatenate([x, conc])
    return x, z





class GAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001):

        #Models
        self.D = None
        self.S = None
        self.G = None

        self.GE = None
        self.SE = None

        self.DM = None
        self.AM = None

        #Config
        self.LR = lr
        self.steps = steps
        self.beta = 0.999

        #Init Models
        self.discriminator()
        self.generator()

        self.GMO = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)
        self.DMO = Adam(lr = self.LR, beta_1 = 0, beta_2 = 0.999)

        self.GE = clone_model(self.G)
        self.GE.set_weights(self.G.get_weights())

        self.SE = clone_model(self.S)
        self.SE.set_weights(self.S.get_weights())

    def discriminator(self):

        if self.D:
            return self.D

        inp = Input(shape = [im_size, im_size, 3])


        x = d_block(inp, 1 * cha)   #128

        x = d_block(x, 2 * cha)   #64

        x = d_block(x, 4 * cha)   #32

        x = d_block(x, 6 * cha)  #16

        x = d_block(x, 8 * cha)  #8

        x = d_block(x, 16 * cha)  #4

        x = d_block(x, 32 * cha, p = False)  #4

        x = Flatten()(x)

        x = Dense(1, kernel_initializer = 'he_uniform')(x)

        self.D = Model(inputs = inp, outputs = x)

        return self.D

    def generator(self):

        if self.G:
            return self.G

        # === Style Mapping ===

        self.S = Sequential()

        self.S.add(Dense(512, input_shape = [latent_size]))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))
        self.S.add(Dense(512))
        self.S.add(LeakyReLU(0.2))


        # === Generator ===

        #Inputs
        inp_style = []

        for i in range(n_layers):
            inp_style.append(Input([512]))

        inp_noise = Input([im_size, im_size, 1])

        #Latent
        x = Lambda(lambda x: x[:, :1] * 0 + 1)(inp_style[0])

        outs = []

        #Actual Model
        x = Dense(4*4*4*cha, activation = 'relu', kernel_initializer = 'random_normal')(x)
        x = Reshape([4, 4, 4*cha])(x)

        x, r = g_block(x, inp_style[0], inp_noise, 32 * cha, u = False)  #4
        outs.append(r)

        x, r = g_block(x, inp_style[1], inp_noise, 16 * cha)  #8
        outs.append(r)

        x, r = g_block(x, inp_style[2], inp_noise, 8 * cha)  #16
        outs.append(r)

        x, r = g_block(x, inp_style[3], inp_noise, 6 * cha)  #32
        outs.append(r)

        x, r = g_block(x, inp_style[4], inp_noise, 4 * cha)   #64
        outs.append(r)

        x, r = g_block(x, inp_style[5], inp_noise, 2 * cha)   #128
        outs.append(r)

        x, r = g_block(x, inp_style[6], inp_noise, 1 * cha)   #256
        outs.append(r)

        x = add(outs)

        x = Lambda(lambda y: y/2 + 0.5)(x) #Use values centered around 0, but normalize to [0, 1], providing better initialization

        self.G = Model(inputs = inp_style + [inp_noise], outputs = x)

        return self.G

    def GenModel(self):

        #Generator Model for Evaluation

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.S(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])

        gf = self.G(style + [inp_noise])

        self.GM = Model(inputs = inp_style + [inp_noise], outputs = gf)

        return self.GM

    def GenModelA(self):

        #Parameter Averaged Generator Model

        inp_style = []
        style = []

        for i in range(n_layers):
            inp_style.append(Input([latent_size]))
            style.append(self.SE(inp_style[-1]))

        inp_noise = Input([im_size, im_size, 1])

        gf = self.GE(style + [inp_noise])

        self.GMA = Model(inputs = inp_style + [inp_noise], outputs = gf)

        return self.GMA

    def EMA(self):

        #Parameter Averaging

        for i in range(len(self.G.layers)):
            up_weight = self.G.layers[i].get_weights()
            old_weight = self.GE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.GE.layers[i].set_weights(new_weight)

        for i in range(len(self.S.layers)):
            up_weight = self.S.layers[i].get_weights()
            old_weight = self.SE.layers[i].get_weights()
            new_weight = []
            for j in range(len(up_weight)):
                new_weight.append(old_weight[j] * self.beta + (1-self.beta) * up_weight[j])
            self.SE.layers[i].set_weights(new_weight)

    def MAinit(self):
        #Reset Parameter Averaging
        self.GE.set_weights(self.G.get_weights())
        self.SE.set_weights(self.S.get_weights())






class StyleGAN(object):

    def __init__(self, steps = 1, lr = 0.0001, decay = 0.00001, silent = True):

        #Init GAN and Eval Models
        self.GAN = GAN(steps = steps, lr = lr, decay = decay)
        self.GAN.GenModel()
        self.GAN.GenModelA()

        self.GAN.G.summary()

        #Data generator (my own code, not from TF 2.0)
        self.im = dataGenerator(directory, im_size, flip = True)

        #Set up variables
        self.lastblip = time.clock()

        self.silent = silent

        self.ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
        self.zeros = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        self.nones = -self.ones

        self.pl_mean = 0
        self.av = np.zeros([44])

    def train(self):

        #Train Alternating
        if random() < mixed_prob:
            style = mixedList(BATCH_SIZE)
        else:
            style = noiseList(BATCH_SIZE)

        #Apply penalties every 16 steps
        apply_gradient_penalty = self.GAN.steps % 2 == 0 or self.GAN.steps < 10000
        apply_path_penalty = self.GAN.steps % 16 == 0

        a, b, c, d = self.train_step(self.im.get_batch(BATCH_SIZE).astype('float32'), style, nImage(BATCH_SIZE), apply_gradient_penalty, apply_path_penalty)

        #Adjust path length penalty mean
        #d = pl_mean when no penalty is applied
        if self.pl_mean == 0:
            self.pl_mean = np.mean(d)
        self.pl_mean = 0.99*self.pl_mean + 0.01*np.mean(d)

        if self.GAN.steps % 10 == 0 and self.GAN.steps > 20000:
            self.GAN.EMA()

        if self.GAN.steps <= 25000 and self.GAN.steps % 1000 == 2:
            self.GAN.MAinit()

        if np.isnan(a):
            print("NaN Value Error.")
            exit()


        #Print info
        if self.GAN.steps % 100 == 0 and not self.silent:
            print("\n\nRound " + str(self.GAN.steps) + ":")
            print("D:", np.array(a))
            print("G:", np.array(b))
            print("PL:", self.pl_mean)

            s = round((time.clock() - self.lastblip), 4)
            self.lastblip = time.clock()

            steps_per_second = 100 / s
            steps_per_minute = steps_per_second * 60
            steps_per_hour = steps_per_minute * 60
            print("Steps/Second: " + str(round(steps_per_second, 2)))
            print("Steps/Hour: " + str(round(steps_per_hour)))

            min1k = floor(1000/steps_per_minute)
            sec1k = floor(1000/steps_per_second) % 60
            print("1k Steps: " + str(min1k) + ":" + str(sec1k))
            steps_left = 200000 - self.GAN.steps + 1e-7
            hours_left = steps_left // steps_per_hour
            minutes_left = (steps_left // steps_per_minute) % 60

            print("Til Completion: " + str(int(hours_left)) + "h" + str(int(minutes_left)) + "m")
            print()

            #Save Model
            if self.GAN.steps % 500 == 0:
                self.save(floor(self.GAN.steps / 10000))
            if self.GAN.steps % 1000 == 0 or (self.GAN.steps % 100 == 0 and self.GAN.steps < 2500):
                self.evaluate(floor(self.GAN.steps / 1000))


        printProgressBar(self.GAN.steps % 100, 99, decimals = 0)

        self.GAN.steps = self.GAN.steps + 1

    @tf.function
    def train_step(self, images, style, noise, perform_gp = True, perform_pl = False):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            #Get style information
            w_space = []
            pl_lengths = self.pl_mean
            for i in range(len(style)):
                w_space.append(self.GAN.S(style[i]))

            #Generate images
            generated_images = self.GAN.G(w_space + [noise])

            #Discriminate
            real_output = self.GAN.D(images, training=True)
            fake_output = self.GAN.D(generated_images, training=True)

            #Hinge loss function
            gen_loss = K.mean(fake_output)
            divergence = K.mean(K.relu(1 + real_output) + K.relu(1 - fake_output))
            disc_loss = divergence

            if perform_gp:
                #R1 gradient penalty
                disc_loss += gradient_penalty(images, real_output, 10)

            if perform_pl:
                #Slightly adjust W space
                w_space_2 = []
                for i in range(len(style)):
                    std = 0.1 / (K.std(w_space[i], axis = 0, keepdims = True) + 1e-8)
                    w_space_2.append(w_space[i] + K.random_normal(tf.shape(w_space[i])) / (std + 1e-8))

                #Generate from slightly adjusted W space
                pl_images = self.GAN.G(w_space_2 + [noise])

                #Get distance after adjustment (path length)
                delta_g = K.mean(K.square(pl_images - generated_images), axis = [1, 2, 3])
                pl_lengths = delta_g

                if self.pl_mean > 0:
                    gen_loss += K.mean(K.square(pl_lengths - self.pl_mean))

        #Get gradients for respective areas
        gradients_of_generator = gen_tape.gradient(gen_loss, self.GAN.GM.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.GAN.D.trainable_variables)

        #Apply gradients
        self.GAN.GMO.apply_gradients(zip(gradients_of_generator, self.GAN.GM.trainable_variables))
        self.GAN.DMO.apply_gradients(zip(gradients_of_discriminator, self.GAN.D.trainable_variables))

        return disc_loss, gen_loss, divergence, pl_lengths

    def evaluate(self, num = 0, trunc = 1.0):

        n1 = noiseList(64)
        n2 = nImage(64)
        trunc = np.ones([64, 1]) * trunc


        generated_images = self.GAN.GM.predict(n1 + [n2], batch_size = BATCH_SIZE)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)
        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+".png")

        # Moving Average

        generated_images = self.GAN.GMA.predict(n1 + [n2, trunc], batch_size = BATCH_SIZE)
        #generated_images = self.generateTruncated(n1, trunc = trunc)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 1))

        c1 = np.concatenate(r, axis = 0)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+"-ema.png")

        #Mixing Regularities
        nn = noise(8)
        n1 = np.tile(nn, (8, 1))
        n2 = np.repeat(nn, 8, axis = 0)
        tt = int(n_layers / 2)

        p1 = [n1] * tt
        p2 = [n2] * (n_layers - tt)

        latent = p1 + [] + p2

        generated_images = self.GAN.GMA.predict(latent + [nImage(64), trunc], batch_size = BATCH_SIZE)
        #generated_images = self.generateTruncated(latent, trunc = trunc)

        r = []

        for i in range(0, 64, 8):
            r.append(np.concatenate(generated_images[i:i+8], axis = 0))

        c1 = np.concatenate(r, axis = 1)
        c1 = np.clip(c1, 0.0, 1.0)

        x = Image.fromarray(np.uint8(c1*255))

        x.save("Results/i"+str(num)+"-mr.png")

    def generateTruncated(self, style, noi = np.zeros([44]), trunc = 0.5, outImage = False, num = 0):

        #Get W's center of mass
        if self.av.shape[0] == 44: #44 is an arbitrary value
            print("Approximating W center of mass")
            self.av = np.mean(self.GAN.S.predict(noise(2000), batch_size = 64), axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        if noi.shape[0] == 44:
            noi = nImage(64)

        w_space = []
        pl_lengths = self.pl_mean
        for i in range(len(style)):
            tempStyle = self.GAN.S.predict(style[i])
            tempStyle = trunc * (tempStyle - self.av) + self.av
            w_space.append(tempStyle)

        generated_images = self.GAN.GE.predict(w_space + [noi], batch_size = BATCH_SIZE)

        if outImage:
            r = []

            for i in range(0, 64, 8):
                r.append(np.concatenate(generated_images[i:i+8], axis = 0))

            c1 = np.concatenate(r, axis = 1)
            c1 = np.clip(c1, 0.0, 1.0)

            x = Image.fromarray(np.uint8(c1*255))

            x.save("Results/t"+str(num)+".png")

        return generated_images

    def saveModel(self, model, name, num):
        json = model.to_json()
        with open("Models/"+name+".json", "w") as json_file:
            json_file.write(json)

        model.save_weights("Models/"+name+"_"+str(num)+".h5")

    def loadModel(self, name, num):

        file = open("Models/"+name+".json", 'r')
        json = file.read()
        file.close()

        mod = model_from_json(json, custom_objects = {'Conv2DMod': Conv2DMod})
        mod.load_weights("Models/"+name+"_"+str(num)+".h5")

        return mod

    def save(self, num): #Save JSON and Weights into /Models/
        self.saveModel(self.GAN.S, "sty", num)
        self.saveModel(self.GAN.G, "gen", num)
        self.saveModel(self.GAN.D, "dis", num)

        self.saveModel(self.GAN.GE, "genMA", num)
        self.saveModel(self.GAN.SE, "styMA", num)


    def load(self, num): #Load JSON and Weights from /Models/

        #Load Models
        self.GAN.D = self.loadModel("dis", num)
        self.GAN.S = self.loadModel("sty", num)
        self.GAN.G = self.loadModel("gen", num)

        self.GAN.GE = self.loadModel("genMA", num)
        self.GAN.SE = self.loadModel("styMA", num)

        self.GAN.GenModel()
        self.GAN.GenModelA()









if __name__ == "__main__":
    model = StyleGAN(lr = 0.0001, silent = False)
    model.evaluate(0)

    while model.GAN.steps < 1000001:
        model.train()

    """
    model.load(31)

    n1 = noiseList(64)
    n2 = nImage(64)
    for i in range(50):
        print(i, end = '\r')
        model.generateTruncated(n1, noi = n2, trunc = i / 50, outImage = True, num = i)
    """














class StarGAN_v2() :
    def __init__(self, sess, args):




















        self.experiment = experiment
        self.phase = args.phase
        self.progressive = args.progressive

        self.model_name = "Climate-StyleGANv2"

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
        self.power_spectra_loss = args.power_spectra_loss
        self.divergence_lambda = 0.001
        self.divergence_loss_flag = False
        self.inference_counter_number = args.inference_counter_number
        self.number_for_l2_images = 200
        self.mode_seeking_gan = False;
        self.channels_list = args.channels_list




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


        ## training with trans indicated should we 
        self.train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}
        self.batch_sizes = get_batch_sizes(self.gpu_num)







        # flags only for styleganv2
        # 
        # 
        self.augment_flag = args.augment_flag
        self.decay_flag = args.decay_flag
        self.decay_iter = args.decay_iter
        self.gan_type = args.gan_type
        self.init_lr = args.lr
        self.ema_decay = args.ema_decay
        self.ch = args.ch
        self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]
        self.c_dim = len(self.label_list)
        self.refer_img_path = args.refer_img_path










        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight
        self.gp_weight = args.gp_weight

        self.sn = args.sn

        """ Generator """
        self.style_dim = args.style_dim
        self.n_layer = args.n_layer
        self.num_style = args.num_style

        """ Discriminator """
        self.n_critic = args.n_critic

        self.img_height = args.img_height
        self.img_width = args.img_width
        self.img_ch = args.img_ch

        




























        # self.model_name = 'StarGAN_v2'
        # self.sess = sess
        # self.phase = args.phase
        # self.checkpoint_dir = args.checkpoint_dir
        # self.sample_dir = args.sample_dir
        # self.result_dir = args.result_dir
        # self.log_dir = args.log_dir
        # self.dataset_name = args.dataset
        # self.dataset_path = os.path.join('./dataset', self.dataset_name)
        # self.augment_flag = args.augment_flag

        # self.decay_flag = args.decay_flag
        # self.decay_iter = args.decay_iter

        # self.gpu_num = args.gpu_num
        # self.iteration = args.iteration // args.gpu_num


        # self.gan_type = args.gan_type

        # self.batch_size = args.batch_size
        # self.print_freq = args.print_freq // args.gpu_num
        # self.save_freq = args.save_freq // args.gpu_num

        # self.init_lr = args.lr
        # self.ema_decay = args.ema_decay
        # self.ch = args.ch

        # self.dataset_path = os.path.join(self.dataset_path, 'train')
        # self.label_list = [os.path.basename(x) for x in glob(self.dataset_path + '/*')]
        # self.c_dim = len(self.label_list)

        # self.refer_img_path = args.refer_img_path

        # """ Weight """
        # self.adv_weight = args.adv_weight
        # self.sty_weight = args.sty_weight
        # self.ds_weight = args.ds_weight
        # self.cyc_weight = args.cyc_weight

        # self.r1_weight = args.r1_weight
        # self.gp_weight = args.gp_weight

        # self.sn = args.sn

        # """ Generator """
        # self.style_dim = args.style_dim
        # self.n_layer = args.n_layer
        # self.num_style = args.num_style

        # """ Discriminator """
        # self.n_critic = args.n_critic

        # self.img_height = args.img_height
        # self.img_width = args.img_width
        # self.img_ch = args.img_ch

        

        print()

        print("##### Information #####")
        print("# gan type : ", self.gan_type)
        print("# selected_attrs : ", self.label_list)
        print("# dataset : ", self.dataset_name)
        print("# batch_size : ", self.batch_size)
        print("# gpu num : ", self.gpu_num)
        print("# iteration : ", self.iteration)
        print("# spectral normalization : ", self.sn)

        print()

        print("##### Generator #####")
        print("# base channel : ", self.ch)
        print("# layer number : ", self.n_layer)

        print()

        print("##### Discriminator #####")
        print("# the number of critic : ", self.n_critic)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, style, scope="generator"):
        channel = self.ch

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) :
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer) :
                x = pre_resblock(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            for i in range(self.n_layer // 2) :
                x = pre_resblock(x, channels=channel, use_bias=True, scope='inter_pre_resblock_' + str(i))

            for i in range(self.n_layer // 2) :
                gamma1 = fully_connected(style, channel, scope='inter_gamma1_fc_' + str(i))
                beta1 = fully_connected(style, channel, scope='inter_beta1_fc_' + str(i))

                gamma2 = fully_connected(style, channel, scope='inter_gamma2_fc_' + str(i))
                beta2 = fully_connected(style, channel, scope='inter_beta2_fc_' + str(i))

                gamma1 = tf.reshape(gamma1, shape=[gamma1.shape[0], 1, 1, -1])
                beta1 = tf.reshape(beta1, shape=[beta1.shape[0], 1, 1, -1])

                gamma2 = tf.reshape(gamma2, shape=[gamma2.shape[0], 1, 1, -1])
                beta2 = tf.reshape(beta2, shape=[beta2.shape[0], 1, 1, -1])

                x = pre_adaptive_resblock(x, channel, gamma1, beta1, gamma2, beta2, use_bias=True, scope='inter_pre_ada_resblock_' + str(i))

            for i in range(self.n_layer) :
                x = up_sample_nearest(x)

                gamma1 = fully_connected(style, channel, scope='up_gamma1_fc_' + str(i))
                beta1 = fully_connected(style, channel, scope='up_beta1_fc_' + str(i))

                gamma2 = fully_connected(style, channel // 2, scope='up_gamma2_fc_' + str(i))
                beta2 = fully_connected(style, channel // 2, scope='up_beta2_fc_' + str(i))

                gamma1 = tf.reshape(gamma1, shape=[gamma1.shape[0], 1, 1, -1])
                beta1 = tf.reshape(beta1, shape=[beta1.shape[0], 1, 1, -1])

                gamma2 = tf.reshape(gamma2, shape=[gamma2.shape[0], 1, 1, -1])
                beta2 = tf.reshape(beta2, shape=[beta2.shape[0], 1, 1, -1])

                x = pre_adaptive_resblock(x, channel // 2, gamma1, beta1, gamma2, beta2, use_bias=True, scope='up_pre_ada_resblock_' + str(i))

                channel = channel // 2

            x = conv(x, channels=self.img_ch, kernel=1, stride=1, use_bias=True, scope='return_image')

            return x

    def style_encoder(self, x_init, scope="style_encoder"):
        channel = self.ch // 2
        style_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer):
                x = pre_resblock_no_norm_relu(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            channel = channel * 2

            for i in range(self.n_layer // 2) :
                x = pre_resblock_no_norm_relu(x, channels=channel, use_bias=True, scope='down_pre_resblock_' + str(i + 4))
                x = down_sample_avg(x)

            kernel_size = int(self.img_height / np.power(2, self.n_layer + self.n_layer // 2))

            x = relu(x)
            x = conv(x, channel, kernel=kernel_size, stride=1, use_bias=True, scope='conv_g_kernel')
            x = relu(x)

            for i in range(self.c_dim) :
                style = fully_connected(x, units=64, use_bias=True, scope='style_fc_' + str(i))
                style_list.append(style)

            return style_list

    def mapping_network(self, latent_z, scope='mapping_network'):
        channel = self.ch * pow(2, self.n_layer)
        style_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = latent_z

            for i in range(self.n_layer + self.n_layer // 2):
                x = fully_connected(x, units=channel, use_bias=True, scope='fc_' + str(i))
                x = relu(x)

            for i in range(self.c_dim) :
                style = fully_connected(x, units=64, use_bias=True, scope='style_fc_' + str(i))
                style_list.append(style)

            return style_list # [c_dim,], style_list[i] = [bs, 64]

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self, x_init, scope="discriminator"):
        channel = self.ch
        logit_list = []
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = conv(x_init, channels=channel, kernel=1, stride=1, use_bias=True, scope='conv_1x1')

            for i in range(self.n_layer):
                x = pre_resblock_no_norm_lrelu(x, channels=channel * 2, use_bias=True, scope='down_pre_resblock_' + str(i))
                x = down_sample_avg(x)

                channel = channel * 2

            channel = channel * 2

            for i in range(self.n_layer // 2):
                x = pre_resblock_no_norm_lrelu(x, channels=channel, use_bias=True, scope='down_pre_resblock_' + str(i + 4))
                x = down_sample_avg(x)

            kernel_size = int(self.img_height / np.power(2, self.n_layer + self.n_layer // 2))

            x = lrelu(x, 0.2)
            x = conv(x, channel, kernel=kernel_size, stride=1, use_bias=True, scope='conv_g_kernel')
            x = lrelu(x, 0.2)

            for i in range(self.c_dim):
                logit = fully_connected(x, units=1, use_bias=True, scope='dis_logit_fc_' + str(i))
                logit_list.append(logit)

            return logit_list

    ##################################################################################
    # Model
    ##################################################################################

    def gradient_panalty(self, real, fake, real_label, scope="discriminator"):
        if self.gan_type.__contains__('dragan'):
            eps = tf.random_uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)

        logits = tf.gather(self.discriminator(interpolated, scope=scope), real_label)


        grad = tf.gradients(logits, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(flatten(grad), axis=-1) # l2 norm

        # WGAN - LP
        GP = 0
        if self.gan_type == 'wgan-lp' :
            GP = self.gp_weight * tf.square(tf.maximum(0.0, grad_norm - 1.))

        elif self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = self.gp_weight * tf.square(grad_norm - 1.)

        return GP

    def build_model(self):

        self.ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay)

        if self.phase == 'train' :
            """ Input Image"""
            img_class = Image_data(self.img_height, self.img_width, self.img_ch, self.dataset_path, self.label_list,
                                   self.augment_flag)
            img_class.preprocess()

            dataset_num = len(img_class.image)
            print("Dataset number : ", dataset_num)

            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            self.ds_weight_placeholder = tf.placeholder(tf.float32, name='ds_weight')


            img_and_label = tf.data.Dataset.from_tensor_slices((img_class.image, img_class.label))

            gpu_device = '/gpu:0'
            img_and_label = img_and_label.apply(shuffle_and_repeat(dataset_num)).apply(
                map_and_batch(img_class.image_processing, self.batch_size * self.gpu_num, num_parallel_batches=16,
                              drop_remainder=True)).apply(prefetch_to_device(gpu_device, None))

            img_and_label_iterator = img_and_label.make_one_shot_iterator()

            self.x_real, label_org = img_and_label_iterator.get_next() # [bs, 256, 256, 3], [bs, 1]
            # label_trg = tf.random_shuffle(label_org)  # Target domain labels
            label_trg = tf.random_uniform(shape=tf.shape(label_org), minval=0, maxval=self.c_dim, dtype=tf.int32) # Target domain labels

            """ split """
            x_real_gpu_split = tf.split(self.x_real, num_or_size_splits=self.gpu_num, axis=0)
            label_org_gpu_split = tf.split(label_org, num_or_size_splits=self.gpu_num, axis=0)
            label_trg_gpu_split = tf.split(label_trg, num_or_size_splits=self.gpu_num, axis=0)

            g_adv_loss_per_gpu = []
            g_sty_recon_loss_per_gpu = []
            g_sty_diverse_loss_per_gpu = []
            g_cyc_loss_per_gpu = []
            g_loss_per_gpu = []

            d_adv_loss_per_gpu = []
            d_loss_per_gpu = []

            for gpu_id in range(self.gpu_num):
                with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):

                        x_real_split = tf.split(x_real_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                        label_org_split = tf.split(label_org_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)
                        label_trg_split = tf.split(label_trg_gpu_split[gpu_id], num_or_size_splits=self.batch_size, axis=0)

                        g_adv_loss = None
                        g_sty_recon_loss = None
                        g_sty_diverse_loss = None
                        g_cyc_loss = None

                        d_adv_loss = None
                        d_simple_gp = None
                        d_gp = None

                        for each_bs in range(self.batch_size) :
                            """ Define Generator, Discriminator """
                            x_real_each = x_real_split[each_bs] # [1, 256, 256, 3]
                            label_org_each = tf.squeeze(label_org_split[each_bs], axis=[0, 1]) # [1, 1] -> []
                            label_trg_each = tf.squeeze(label_trg_split[each_bs], axis=[0, 1])

                            random_style_code = tf.random_normal(shape=[1, self.style_dim])
                            random_style_code_1 = tf.random_normal(shape=[1, self.style_dim])
                            random_style_code_2 = tf.random_normal(shape=[1, self.style_dim])

                            random_style = tf.gather(self.mapping_network(random_style_code), label_trg_each)
                            random_style_1 = tf.gather(self.mapping_network(random_style_code_1), label_trg_each)
                            random_style_2 = tf.gather(self.mapping_network(random_style_code_2), label_trg_each)

                            x_fake = self.generator(x_real_each, random_style) # for adversarial objective
                            x_fake_1 = self.generator(x_real_each, random_style_1) # for style diversification
                            x_fake_2 = self.generator(x_real_each, random_style_2) # for style diversification

                            x_real_each_style = tf.gather(self.style_encoder(x_real_each), label_org_each) # for cycle consistency
                            x_fake_style = tf.gather(self.style_encoder(x_fake), label_trg_each) # for style reconstruction

                            x_cycle = self.generator(x_fake, x_real_each_style) # for cycle consistency

                            real_logit = tf.gather(self.discriminator(x_real_each), label_org_each)
                            fake_logit = tf.gather(self.discriminator(x_fake), label_trg_each)

                            """ Define loss """
                            if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan':
                                GP = self.gradient_panalty(real=x_real_each, fake=x_fake, real_label=label_org_each)
                            else:
                                GP = tf.constant([0], tf.float32)

                            if each_bs == 0 :
                                g_adv_loss = self.adv_weight * generator_loss(self.gan_type, fake_logit)
                                g_sty_recon_loss = self.sty_weight * L1_loss(random_style, x_fake_style)
                                g_sty_diverse_loss = self.ds_weight_placeholder * L1_loss(x_fake_1, x_fake_2)
                                g_cyc_loss = self.cyc_weight * L1_loss(x_real_each, x_cycle)

                                d_adv_loss = self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)
                                d_simple_gp = self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)
                                d_gp = self.adv_weight * GP

                            else :
                                g_adv_loss = tf.concat([g_adv_loss, self.adv_weight * generator_loss(self.gan_type, fake_logit)], axis=0)
                                g_sty_recon_loss = tf.concat([g_sty_recon_loss, self.sty_weight * L1_loss(random_style, x_fake_style)], axis=0)
                                g_sty_diverse_loss = tf.concat([g_sty_diverse_loss, self.ds_weight_placeholder * L1_loss(x_fake_1, x_fake_2)], axis=0)
                                g_cyc_loss = tf.concat([g_cyc_loss, self.cyc_weight * L1_loss(x_real_each, x_cycle)], axis=0)

                                d_adv_loss = tf.concat([d_adv_loss, self.adv_weight * discriminator_loss(self.gan_type, real_logit, fake_logit)], axis=0)
                                d_simple_gp = tf.concat([d_simple_gp, self.adv_weight * simple_gp(real_logit, fake_logit, x_real_each, x_fake, r1_gamma=self.r1_weight, r2_gamma=0.0)], axis=0)
                                d_gp = tf.concat([d_gp, self.adv_weight * GP], axis=0)


                        g_adv_loss = tf.reduce_mean(g_adv_loss)
                        g_sty_recon_loss = tf.reduce_mean(g_sty_recon_loss)
                        g_sty_diverse_loss = tf.reduce_mean(g_sty_diverse_loss)
                        g_cyc_loss = tf.reduce_mean(g_cyc_loss)

                        d_adv_loss = tf.reduce_mean(d_adv_loss)
                        d_simple_gp = tf.reduce_mean(tf.reduce_sum(d_simple_gp, axis=[1, 2, 3]))
                        d_gp = tf.reduce_mean(d_gp)

                        g_loss = g_adv_loss + g_sty_recon_loss - g_sty_diverse_loss + g_cyc_loss
                        d_loss = d_adv_loss + d_simple_gp + d_gp

                        g_adv_loss_per_gpu.append(g_adv_loss)
                        g_sty_recon_loss_per_gpu.append(g_sty_recon_loss)
                        g_sty_diverse_loss_per_gpu.append(g_sty_diverse_loss)
                        g_cyc_loss_per_gpu.append(g_cyc_loss)

                        d_adv_loss_per_gpu.append(d_adv_loss)

                        g_loss_per_gpu.append(g_loss)
                        d_loss_per_gpu.append(d_loss)

            g_adv_loss = tf.reduce_mean(g_adv_loss_per_gpu)
            g_sty_recon_loss = tf.reduce_mean(g_sty_recon_loss_per_gpu)
            g_sty_diverse_loss = tf.reduce_mean(g_sty_diverse_loss_per_gpu)
            g_cyc_loss = tf.reduce_mean(g_cyc_loss_per_gpu)
            self.g_loss = tf.reduce_mean(g_loss_per_gpu)

            d_adv_loss = tf.reduce_mean(d_adv_loss_per_gpu)
            self.d_loss = tf.reduce_mean(d_loss_per_gpu)


            """ Training """
            t_vars = tf.trainable_variables()
            G_vars = [var for var in t_vars if 'generator' in var.name]
            E_vars = [var for var in t_vars if 'encoder' in var.name]
            F_vars = [var for var in t_vars if 'mapping' in var.name]
            D_vars = [var for var in t_vars if 'discriminator' in var.name]

            if self.gpu_num == 1 :
                prev_g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=G_vars)
                prev_e_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=E_vars)
                prev_f_optimizer = tf.train.AdamOptimizer(self.lr * 0.01, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=F_vars)

                self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.d_loss, var_list=D_vars)

            else :
                prev_g_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss, var_list=G_vars,
                                                                                                 colocate_gradients_with_ops=True)
                prev_e_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.g_loss,
                                                                                                 var_list=E_vars,
                                                                                                 colocate_gradients_with_ops=True)
                prev_f_optimizer = tf.train.AdamOptimizer(self.lr * 0.01, beta1=0, beta2=0.99).minimize(self.g_loss,
                                                                                                        var_list=F_vars,
                                                                                                        colocate_gradients_with_ops=True)

                self.d_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0, beta2=0.99).minimize(self.d_loss,
                                                                                                 var_list=D_vars,
                                                                                                 colocate_gradients_with_ops=True)

            with tf.control_dependencies([prev_g_optimizer, prev_e_optimizer, prev_f_optimizer]):
                self.g_optimizer = self.ema.apply(G_vars)
                self.e_optimizer = self.ema.apply(E_vars)
                self.f_optimizer = self.ema.apply(F_vars)

            """" Summary """
            self.Generator_loss = tf.summary.scalar("g_loss", self.g_loss)
            self.Discriminator_loss = tf.summary.scalar("d_loss", self.d_loss)

            self.g_adv_loss = tf.summary.scalar("g_adv_loss", g_adv_loss)
            self.g_sty_recon_loss = tf.summary.scalar("g_sty_recon_loss", g_sty_recon_loss)
            self.g_sty_diverse_loss = tf.summary.scalar("g_sty_diverse_loss", g_sty_diverse_loss)
            self.g_cyc_loss = tf.summary.scalar("g_cyc_loss", g_cyc_loss)

            self.d_adv_loss = tf.summary.scalar("d_adv_loss", d_adv_loss)

            g_summary_list = [self.Generator_loss, self.g_adv_loss, self.g_sty_recon_loss, self.g_sty_diverse_loss, self.g_cyc_loss]
            d_summary_list = [self.Discriminator_loss, self.d_adv_loss]

            self.g_summary_loss = tf.summary.merge(g_summary_list)
            self.d_summary_loss = tf.summary.merge(d_summary_list)

            """ Result Image """
            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.x_fake_list = []
            first_x_real = tf.expand_dims(self.x_real[0], axis=0)

            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            for _ in range(self.num_style):
                random_style_code = tf.truncated_normal(shape=[1, self.style_dim])
                self.x_fake_list.append(tf.map_fn(
                    lambda c: return_g_images(self.generator,
                                              first_x_real,
                                              tf.gather(self.mapping_network(random_style_code), c)),
                    label_fix_list, dtype=tf.float32))

        elif self.phase == 'refer_test':
            """ Test """

            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')
            self.refer_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='refer_image')


            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            self.refer_fake_image = tf.map_fn(
                lambda c : return_g_images(self.generator,
                                           self.custom_image,
                                           tf.gather(self.style_encoder(self.refer_image), c)),
                label_fix_list, dtype=tf.float32)

        else :
            """ Test """

            def return_g_images(generator, image, code):
                x = generator(image, code)
                return x

            self.custom_image = tf.placeholder(tf.float32, [1, self.img_height, self.img_width, self.img_ch], name='custom_image')
            label_fix_list = tf.constant([idx for idx in range(self.c_dim)])

            random_style_code = tf.truncated_normal(shape=[1, self.style_dim])
            self.custom_fake_image = tf.map_fn(
                lambda c : return_g_images(self.generator,
                                           self.custom_image,
                                           tf.gather(self.mapping_network(random_style_code), c)),
                label_fix_list, dtype=tf.float32)



    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=10)

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_batch_id = checkpoint_counter
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        # loop for epoch
        start_time = time.time()
        past_g_loss = -1.
        lr = self.init_lr
        ds_w = self.ds_weight

        for idx in range(start_batch_id, self.iteration):
            if self.decay_flag :
                total_step = self.iteration
                current_step = idx
                decay_start_step = self.decay_iter

                if current_step < decay_start_step :
                    # lr = self.init_lr
                    ds_w = self.ds_weight
                else :
                    # lr = self.init_lr * (total_step - current_step) / (total_step - decay_start_step)
                    ds_w = self.ds_weight * (total_step - current_step) / (total_step - decay_start_step)

                """ half decay """
                """
                if idx > 0 and (idx % decay_start_step) == 0 :
                    lr = self.init_lr * pow(0.5, idx // decay_start_step)
                """

            train_feed_dict = {
                self.lr : lr,
                self.ds_weight_placeholder: ds_w
            }

            # Update D
            _, d_loss, summary_str = self.sess.run([self.d_optimizer, self.d_loss, self.d_summary_loss], feed_dict = train_feed_dict)
            self.writer.add_summary(summary_str, counter)

            # Update G
            g_loss = None
            if (counter - 1) % self.n_critic == 0 :
                real_images, fake_images, _, _, _, g_loss, summary_str = self.sess.run([self.x_real, self.x_fake_list,
                                                                                  self.g_optimizer, self.e_optimizer, self.f_optimizer,
                                                                                  self.g_loss, self.g_summary_loss], feed_dict = train_feed_dict)
                self.writer.add_summary(summary_str, counter)
                past_g_loss = g_loss

            # display training status
            counter += 1
            if g_loss == None :
                g_loss = past_g_loss

            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (idx, self.iteration, time.time() - start_time, d_loss, g_loss))

            if np.mod(idx+1, self.print_freq) == 0 :
                real_image = np.expand_dims(real_images[0], axis=0)
                save_images(real_image, [1, 1],
                            './{}/real_{:07d}.jpg'.format(self.sample_dir, idx+1))

                merge_fake_x = None

                for ns in range(self.num_style) :
                    fake_img = np.transpose(fake_images[ns], axes=[1, 0, 2, 3, 4])[0]

                    if ns == 0 :
                        merge_fake_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                    else :
                        x = return_images(fake_img, [1, self.c_dim])
                        merge_fake_x = np.concatenate([merge_fake_x, x], axis=0)

                merge_fake_x = np.expand_dims(merge_fake_x, axis=0)
                save_images(merge_fake_x, [1, 1],
                            './{}/fake_{:07d}.jpg'.format(self.sample_dir, idx+1))

            if np.mod(counter - 1, self.save_freq) == 0:
                self.save(self.checkpoint_dir, counter)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):

        if self.sn:
            sn = '_sn'
        else:
            sn = ''

        return "{}_{}_{}_{}adv_{}sty_{}ds_{}cyc{}".format(self.model_name, self.dataset_name, self.gan_type,
                                                          self.adv_weight, self.sty_weight, self.ds_weight, self.cyc_weight,
                                                          sn)

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
        test_files = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'test')) + glob('./dataset/{}/{}/*.png'.format(self.dataset_name, 'test'))

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name or 'encoder' in var.name or 'mapping' in var.name]

        shadow_G_vars_dict = {}

        for g_var in G_vars:
            shadow_G_vars_dict[self.ema.average_name(g_var)] = g_var

        self.saver = tf.train.Saver(shadow_G_vars_dict)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(test_files):
            print("Processing image: " + sample_file)
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

            merge_x = None

            for i in range(self.num_style) :
                fake_img = self.sess.run(self.custom_fake_image, feed_dict={self.custom_image: sample_image})
                fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]

                if i == 0:
                    merge_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
                else :
                    x = return_images(fake_img, [1, self.c_dim])
                    merge_x = np.concatenate([merge_x, x], axis=0)

            merge_x = np.expand_dims(merge_x, axis=0)

            save_images(merge_x, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../..' + os.path.sep + sample_file), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../..' + os.path.sep + image_path), self.img_width * self.c_dim, self.img_height * self.num_style))
            index.write("</tr>")

        index.close()

    def refer_test(self):
        tf.global_variables_initializer().run()
        test_files = glob('./dataset/{}/{}/*.jpg'.format(self.dataset_name, 'test')) + glob('./dataset/{}/{}/*.png'.format(self.dataset_name, 'test'))

        refer_image = load_test_image(self.refer_img_path, self.img_width, self.img_height, self.img_ch)

        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name or 'encoder' in var.name or 'mapping' in var.name]

        shadow_G_vars_dict = {}

        for g_var in G_vars:
            shadow_G_vars_dict[self.ema.average_name(g_var)] = g_var

        self.saver = tf.train.Saver(shadow_G_vars_dict)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        self.result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(self.result_dir)

        self.result_dir = os.path.join(self.result_dir, 'refer_results')
        check_folder(self.result_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file in tqdm(test_files):
            print("Processing image: " + sample_file)
            sample_image = load_test_image(sample_file, self.img_width, self.img_height, self.img_ch)
            image_path = os.path.join(self.result_dir, '{}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.refer_fake_image, feed_dict={self.custom_image: sample_image, self.refer_image: refer_image})
            fake_img = np.transpose(fake_img, axes=[1, 0, 2, 3, 4])[0]

            merge_x = return_images(fake_img, [1, self.c_dim]) # [self.img_height, self.img_width * self.c_dim, self.img_ch]
            merge_x = np.expand_dims(merge_x, axis=0)

            save_images(merge_x, [1, 1], image_path)

            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                        '../../..' + os.path.sep + sample_file), self.img_width, self.img_height))

            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                        '../../..' + os.path.sep + image_path), self.img_width * self.c_dim, self.img_height))
            index.write("</tr>")

        index.close()
