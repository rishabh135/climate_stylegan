from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from StyleGAN import StyleGAN


import argparse
from utils import *
from datetime import datetime

# from comet_ml import Experiment
# from comet_ml import OfflineExperiment


import wandb as experiment


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of StyleGAN for climate data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train, test, draw]')
    parser.add_argument('--draw', type=str, default='style_mix', help='[uncurated, style_mix, truncation_trick]')
    parser.add_argument('--dataset', type=str, default= "climate_3000", help='The dataset name what you want to generate')
    parser.add_argument('--iteration', type=int, default=120, help='The number of images used in the train phase 120k by default')
    parser.add_argument('--max_iteration', type=int, default=2500, help='The total number of images 2500k by default')

    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch in the test phase')
    parser.add_argument('--gpu_num', type=int, default=8, help='The number of gpu')
    parser.add_argument('--progressive', type=str2bool, default=True, help='use progressive training')
    parser.add_argument('--sn', type=str2bool, default=False, help='use spectral normalization')

    parser.add_argument('--start_res', type=int, default=8, help='The number of starting resolution')

    parser.add_argument('--climate_img_size', type=int, default=512, help='Size of original climate data 512 by default')

    parser.add_argument('--img_size', type=int, default=256, help='The target size of image')

    parser.add_argument('--test_num', type=int, default=100, help='The number of generating images in the test phase')
    parser.add_argument('--input_channels', type=int, default=1, help='The number of input channels for the input real images')
    parser.add_argument('--seed', type=str2bool, default=True, help='seed in the draw phase')

    parser.add_argument('--checkpoint_dir', type=str, default='./stored_outputs/wo_norm/checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='./stored_outputs/wo_norm/results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='./stored_outputs/wo_norm/logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='./stored_outputs/wo_norm/samples',
                        help='Directory name to save the samples on training')


    parser.add_argument('--divergence_loss_flag', type=bool, default= False,
                        help='should there be added term to the g_loss from divergence , default is false')

    parser.add_argument('--power_spectra_loss', type=bool, default= False,
                        help='whether to add a power loss term to the generator loss, default is false')

    parser.add_argument('--style_mixing_flag', type=bool, default= False,
                        help='should there be style mixing of two latents from g_mapping network , default is false')


    parser.add_argument('--dataset_location', type=str, default="/global/cscratch1/sd/rgupta2/backup/climate_stylegan/dataset/climate_data_original/",
                        help='dataset_directory')



    parser.add_argument('--name_experiment', type=str, default="climate_data_wo_norm",
                        help='name of the experiment, on comet_ml')


    parser.add_argument('--inference_counter_number', type=int, default=0,
                    help='model_number_loaded_in_inference_stage')


    parser.add_argument('--channels_list', type=list, default=None,
                    help='subset_of_channels_to_be_used_for_training')


    parser.add_argument('--crop_size', type=int, default=512,
                    help='crop_size_of_orginal_image_for_data_processing')

    parser.add_argument('--tsne_real_data', type=bool, default= False,
                        help='discriminator_tsne embedding for real data, default is false')


    parser.add_argument('--only_ux', type=bool, default= False,
                        help='only ux for training, default is false')

    parser.add_argument('--both_ux_uy', type=bool, default= False,
                        help='both ux and uy for training, default is false')


    parser.add_argument('--custom_cropping_flag', type=bool, default= False,
                        help='cropping input images across the equator with y axis fixed and x axis varied')

    parser.add_argument('--fixed_offset', type=int, default= -1,
                        help='default value to be used for custom cropping between 0 and 383 pixels or will select randomly')

    
    parser.add_argument('--decay_logan', type=bool, default= True,
                        help='annealing of logan for better tail fit')

    parser.add_argument('--logan_flag', type=bool, default= True,
                        help='whether to train without logan')
    

    parser.add_argument('--feature_matching_loss', type=bool, default= True,
                        help='whether to use feature matching loss from Improving GAN paper by Tim Salimans')

    
    parser.add_argument('--wandb_flag', type=bool, default= True,
                        help='Flag to log with wandb or not')

    parser.add_argument('--tanh_flag', type=bool, default= False,
                        help='Flag to use tanh in the last layer of generator or not')

    parser.add_argument('--featuremap_factor', type=int, default= 1,
                        help='how much the featuremaps should be divided by')
    
    return check_args(parser.parse_args())




# ["pr", "prw", "psl", "ts", "ua850", "va850", "omega"] the channels order

"""checking arguments"""
def check_args(args):

    # import comet_ml in the top of your file


    # experiment = Experiment(api_key="YC7c0hMcGsJyRRjD98waGBcVa", project_name="{}_{}".format("logan_climategan_norm_chanel_", args.input_channels), workspace="style-gan")


    # experiment = Experiment(api_key="lsfFN2N0VlRIMOwAg9rmJ2SAf", project_name="{}".format(args.name_experiment), workspace="style-gan")

  

    # experiment = OfflineExperiment(project_name="{}_{}".format("args.name_experiment", args.input_channels), workspace="style-gan" ,offline_directory="./comet_ml_offline_experiments/feature_matching_step_annealed_logan/")




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
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()


    hyper_params = vars(args)
    # experiment.config(hyper_params)


    experiment.init(project="stylegan-v1-tf-{}".format(args.name_experiment), name="Run from {}".format(datetime.now().strftime('%H:%M-%d-%B-%Y')),  dir="/global/cscratch1/sd/rgupta2/backup/climate_stylegan/wandb_data/", config= hyper_params )
      

        # experiment = Experiment(api_key="lsfFN2N0VlRIMOwAg9rmJ2SAf", project_name="{}".format(args.name_experiment), workspace="style-gan")        


    if args is None:
        exit()

    # experiment.set_name(args.name_experiment)
    # open session
    # Assume that you have 12GB of GPU memory and want to allocate ~4GB:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99, allow_growth = True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        # with experiment.train():
        gan = StyleGAN(sess, args, experiment)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        # experiment.set_model_graph(sess.graph)

        if args.phase == 'train' :
            # launch the graph in a session
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")


        if args.phase == 'discriminator_tsne' :
            gan.discriminator_tsne()
            print(" [*] discriminator_tsne finished!")

        
        if args.phase == 'interpolate_inference' :
            gan.interpolate_inference()
            print(" [*] interpolate_inference finished!")

        if args.phase == 'draw' :
            if args.draw == 'style_mix' :
                gan.draw_style_mixing_figure()
                print(" [*] Style mix finished!")

            elif args.draw == 'truncation_trick' :
                gan.draw_truncation_trick_figure()
                print(" [*] Truncation_trick finished!")

            else :
                gan.draw_uncurated_result_figure()
                print(" [*] Un-curated finished!")

if __name__ == '__main__':
    main()






