import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import fftpack
from cycler import cycler


def pixhist(imgs, vals, inverse_transf=None):
    if inverse_transf:
        imgs = inverse_transf(imgs)
    val_hist, val_bin_edges = np.histogram(vals, bins=100 , range=(0.0001, 0.21) )
    gen_hist, gen_bin_edges  = np.histogram(imgs, bins=100, range=(0.0001, 0.21))
    
    val_centers = (val_bin_edges[:-1] + val_bin_edges[1:]) / 2
    
    # gen_centers = (gen_bin_edges[:-1] + gen_bin_edges[1:]) / 2
    

    fig = plt.figure(figsize = (8,10), dpi=200)
    plt.errorbar(val_centers, val_hist, yerr=np.sqrt(val_hist), fmt='ks--', label='real')
    plt.errorbar(val_centers, gen_hist, yerr=np.sqrt(gen_hist), fmt='ro', label='generated')
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.legend()
    sqdiff = np.power(val_hist - gen_hist, 2.0)
    
    val_hist[val_hist<=0.] = 1.

#     plt.savefig("./normalized_prec_data_0.008_at_generator_87500.jpeg", dpi=200)
    return fig, np.sum(np.divide(sqdiff, val_hist))


def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged power spectrum (1D), for a batch of images.
    image - The image tensor, [N,C,H,W] format
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fractional pixels).
    
    """
    batch, channel, height, width = image.shape
    # Calculate the indices from the image
    y, x = np.indices([height, width])
    
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    ind = np.argsort(r.flat)

    # Get sorted radii
    r_sorted = r.flat[ind]
    i_sorted = np.reshape(image, (batch, channel, -1,))[:,:,ind]
    
    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(np.int32)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    
    csum = np.cumsum(i_sorted, axis=-1)
    tbin = csum[:,:,rind[1:]] - csum[:,:,rind[:-1]]
    radial_prof = tbin / nr

    return radial_prof


def power_spectrum(image):
    """Computes azimuthal average of 2D power spectrum of a np array image batch.
       For plotting power spectra of images against validation set."""
    GLOBAL_MEAN = 1. 
    F1 = fftpack.fftn((image - GLOBAL_MEAN)/GLOBAL_MEAN, axes=[2,3])
    F2 = fftpack.fftshift(F1, axes=[2,3])
    pspec2d = np.abs(F2)**2
    P_k = np.squeeze(azimuthalAverage(pspec2d))
    k = np.arange(P_k.shape[1])
    return k, P_k


def pspect(imgs, vals, inverse_transf=None):
    if inverse_transf:
        imgs = inverse_transf(imgs)
    k, Pk_val = power_spectrum(vals)
    k, Pk_gen = power_spectrum(imgs)
    val_mean = np.mean(Pk_val, axis=0)
    gen_mean = np.mean(Pk_gen, axis=0)
    val_std = np.std(Pk_val, axis=0)
    gen_std = np.std(Pk_gen, axis=0)


    fig = plt.figure(figsize = (8,10), dpi=200)
    # print(" \n ********************  inside  pspect gen_mean.shape {} and gen-std.shape {} *****************\n\n ".format(gen_mean.shape, gen_std.shape ))
    plt.fill_between(k, (gen_mean - gen_std).squeeze(), (gen_mean + gen_std).squeeze(), color='red', alpha=0.4)
    plt.plot(k, val_mean, 'k:')
    plt.plot(k, gen_mean, 'r--')
    plt.plot(k, val_mean + val_std, 'k-')
    plt.plot(k, val_mean - val_std, 'k-')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')
    plt.title('Power Spectrum')
    chi_val = np.sum(np.divide(np.power(gen_mean[:64] - val_mean[:64], 2.0), val_mean[:64]))
    print(" chi val : {} ".format(chi_val))
    return fig, chi_val




def pspect_group(validation_images, fig, idx, figsize_1, figsize_2, inverse_transf=None):
    if inverse_transf:
        imgs = inverse_transf(imgs)

    # fig = plt.figure(figsize = (8,10), dpi=200)


    # 1. Setting prop cycle on default rc parameter
    plt.rc('lines', linewidth=4)
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'saddlebrown', 'b', 'g' , 'saddlebrown']) + cycler('linestyle', ['-', '--', ':', '-.', '-'])))

    fig.add_subplot( figsize_1, figsize_2,  idx)
    idx += 1
    labels = ["src_image", "src_image_latent_coarse", "src_image_latent_middle", "src_image_latent_fine", "dst_image" ]
    for i, image in enumerate(validation_images):

        print("\n image shape before : {} ".format(image.shape))
        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = np.repeat(image, 100, axis=0)
        print("image shape after : {} \n".format(image.shape))
        k, Pk_val = power_spectrum(image)
        val_mean = np.mean(Pk_val, axis=0)
        val_std = np.std(Pk_val, axis=0)
        plt.plot(k, val_mean , label = labels[i],  linewidth=0.5)

    # plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'$P(k)$')
    plt.xlabel(r'$k$')


    # leg = plt.legend(loc="upper right", bbox_to_anchor=[0, 1], shadow=True, title="Legends", fancybox=True)


    # leg = plt.legend(loc='best')
    # # get the lines and texts inside legend box
    # leg_lines = leg.get_lines()
    # leg_texts = leg.get_texts()
    # # bulk-set the properties of all lines and texts
    # plt.setp(leg_lines, linewidth=4)
    # plt.setp(leg_texts, fontsize='small')


    plt.title('Power Spectrum')


    # plt.savefig( '{}/custom-style-mixing_with_radial_profile.jpg'.format(result_dir) , bbox_inches='tight', dpi = 400)
    return 

