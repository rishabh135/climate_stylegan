import tensorflow as tf
import numpy as np


from random import seed
from random import randint
# seed random number generator
seed(1)


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


def list_slice(tensor, indices, axis):
    """
    Args
    ----
    tensor (Tensor) : input tensor to slice
    indices ( [int] ) : list of indices of where to perform slices
    axis (int) : the axis to perform the slice on
    """

    slices = []   

    ## Set the shape of the output tensor. 
    # Set any unknown dimensions to -1, so that reshape can infer it correctly. 
    # Set the dimension in the slice direction to be 1, so that overall dimensions are preserved during the operation
    shape = tensor.get_shape().as_list()
    shape[shape==None] = -1
    shape[axis] = 1

    nd = len(shape)

    for i in indices:   
        _slice = [slice(None)]*nd
        _slice[axis] = slice(i,i+1)
        slices.append(tf.reshape(tensor[_slice], shape))

    return tf.concat(slices, axis=axis)


def parse_fn(img, res, input_channels, img_size, dtype, channels_list, crop_size, only_ux, both_ux_uy, custom_cropping_flag, climate_img_size, fixed_offset):
    img = tf.decode_raw(img, dtype)
    img = tf.reshape(img, [img_size, img_size, -1]) 


    print(" original image size : {} crop size {} ".format(img.shape, crop_size ))
    offset_height= (climate_img_size-crop_size)//2

    if(fixed_offset > 0):
        offset_width = fixed_offset

    else:
        offset_width = randint(0, climate_img_size- crop_size-1)

    target_height = crop_size
    target_width = crop_size

    #  (self.climate_img_size - self.crop_size)//2    (self.climate_img_size - self.crop_size)//2 + self.crop_size

    if(custom_cropping_flag == True):
        img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, target_height, target_width)

    else:
        img = tf.image.resize(img, size=[res, res], method=tf.image.ResizeMethod.BILINEAR)
    

    """
    To crop a image randomly from the given dataset of shape 512

    """
    # img  = tf.random_crop(img, size=(input_channels, crop_size, crop_size))
    
    # img = tf.transpose(img, perm=[1,2,0])
    
    if(only_ux):
        img = img[:,:,4:5]
    elif(both_ux_uy):
        img = img[:,:,4:6]
    else:
        img = img[:,:,-input_channels:]
    # if(channels_list != None):
    #     img = list_slice(img, channels_list, 2)
    
    return img

def build_input_pipeline(filelist, res, batch_size, gpu_device, input_channels, channels_list, crop_size, only_ux=False, both_ux_uy=False, repeat_flag=True, custom_cropping_flag=False, climate_img_size=512, fixed_offset = -1):

    with tf.device('/cpu:0'):
        npy_file = filelist[0]
        shape = np.load(npy_file, mmap_mode='r')[0].shape
        num_features = np.product(shape)
        dtype = tf.float32
        header_offset = npy_header_offset(npy_file)

        dataset = tf.data.FixedLengthRecordDataset(filelist,num_features*dtype.size, header_bytes=header_offset)
        
        dataset = dataset.map(lambda img: parse_fn(img, res, input_channels, shape[1], dtype, channels_list, crop_size, only_ux, both_ux_uy, custom_cropping_flag, climate_img_size, fixed_offset),
                                                    num_parallel_calls=4)

        if(repeat_flag):
            dataset = dataset.repeat(None)
        dataset = dataset.shuffle(batch_size*100)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(10)
        inputs = dataset.apply(tf.data.experimental.prefetch_to_device(gpu_device, None))

        inputs_iterator = inputs.make_one_shot_iterator()
        batch = inputs_iterator.get_next()
        batch = tf.reshape(batch, (batch_size, res, res, input_channels))
        return batch
