import tensorflow as tf
import numpy as np

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

def parse_fn(img, res, input_channels, img_size, dtype):
  img = tf.decode_raw(img, dtype)
  img = tf.reshape(img, [input_channels, img_size, img_size])
  img = tf.transpose(img, perm=[1,2,0])
  return tf.image.resize(img, size=[res, res], method=tf.image.ResizeMethod.BILINEAR)

def build_input_pipeline(filelist, res, batch_size, gpu_device):

  with tf.device('/cpu:0'):
    npy_file = filelist[0]
    shape = np.load(npy_file, mmap_mode='r')[0].shape
    num_features = np.product(shape)
    dtype = tf.float32
    header_offset = npy_header_offset(npy_file)

    dataset = tf.data.FixedLengthRecordDataset(filelist,
                                               num_features*dtype.size,
                                               header_bytes=header_offset)
    
    dataset = dataset.map(lambda img: parse_fn(img, res, shape[0], shape[1], dtype),
                          num_parallel_calls=4)

    dataset = dataset.repeat(None)
    dataset = dataset.shuffle(batch_size*100)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(10)
    inputs = dataset.apply(tf.data.experimental.prefetch_to_device(gpu_device, None))

    inputs_iterator = inputs.make_one_shot_iterator()
    batch = inputs_iterator.get_next()
    batch = tf.reshape(batch, (batch_size, res, res, shape[0]))

    return batch

