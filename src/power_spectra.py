import numpy as np
import tensorflow as tf

def get_indices(shape, center=None):

    batch, channel, height, width = shape

    # Calculate the indices from the image
    y, x = np.indices([height, width])
    y = np.tile(y, (batch, channel, 1, 1))
    x = np.tile(x, (batch, channel, 1, 1))

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    return tf.constant(r)

def azimAvg_tensor(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The image tensor, [N,C,H,W] format
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    batch, channel, height, width = image.shape.as_list()
    r = get_indices(image.shape.as_list(), center)

    # Get sorted radii
    ind = tf.argsort(tf.reshape(r, (batch, channel, -1,)))
    r_sorted = tf.gather(tf.reshape(r, (batch, channel, -1,)), ind, batch_dims=2)
    i_sorted = tf.gather(tf.reshape(image, (batch, channel, -1,)), ind, batch_dims=2)


    # Get the integer part of the radii (bin size = 1)
    r_int = tf.cast(r_sorted, tf.int32)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[:,:,1:] - r_int[:,:,:-1]  # Assumes all radii represented
    rind = tf.reshape(tf.where(deltar)[:,2], (batch, -1))    # location of changes in radius
    rind = tf.expand_dims(rind, axis=1)
    nr = tf.cast(rind[:,:,1:] - rind[:,:,:-1], tf.float32)        # number of radius bin

    # Cumulative sum to figure out sums for each radius bin

    csum = tf.cumsum(i_sorted, axis=-1)
    tbin = tf.gather(csum, rind[:,:,1:], batch_dims=2) - tf.gather(csum, rind[:,:,:-1], batch_dims=2)
    radial_prof = tbin / nr

    return radial_prof


def batch_power_spectrum(image):
    shuffled = tf.transpose(tf.cast(image, dtype=tf.complex64), perm=(0,3,1,2))
    F1 = tf.signal.fft2d(shuffled)
    F2 = tf.signal.fftshift(F1, axes=(2,3))
    #pspec2d = tf.abs(F2)**2
    pspec2d = tf.abs(F2)
    P_k = azimAvg_tensor(pspec2d)

    return tf.squeeze(P_k)
