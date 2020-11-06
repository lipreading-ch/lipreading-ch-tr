import os
import time
from functools import partial
import tensorflow as tf

VIDEO_LENGTH = 200
LABEL_MAX_LENGTH = 250
VIDEO_FRAME_SHAPE = (64, 128)
EXAMPLE_FEATURES = {

    'video': tf.VarLenFeature(tf.string),
    'pinyin': tf.FixedLenFeature([], tf.string),
    'tone': tf.FixedLenFeature([], tf.string),
    'viseme': tf.VarLenFeature(tf.int64),
    'word': tf.VarLenFeature(tf.int64),
}

JITTER_P = 0.05
FLIP_P = 0.5


def read_images(images_raw, size, channel, aug):
    """ Read raw images to tensor.
        For example T raw image will be read to Tensor of
        shape (T, h, w, channel)

    Args:
        images_raw: 1-d `string` Tensor. Each element is an encoded jpeg image.
        size: Tuple (h, w).  The image will be resized to such size.
        channel: Int. 1 will output grayscale images, 3 outputs RGB
                 images.
        aug: Boolean. Add frame jitter and random flip if aug is True.

    Returns: 4-D `float32` Tensor. The decoded images.
    """

    i = tf.constant(0)
    j = tf.constant(0)
    image_length = tf.shape(images_raw)[0]
    images = tf.TensorArray(
        dtype=tf.float32, size=image_length, dynamic_size=True)

    condition = lambda i, j, images: tf.less(i, image_length)

    flip_prob = tf.random_uniform([], minval=0, maxval=1)

    def loop_body(i, j, images):
        """ The loop body of reading images.
        """
        image = tf.image.resize_images(
            tf.image.decode_jpeg(images_raw[i], channels=channel),
            size=size,
            method=tf.image.ResizeMethod.BILINEAR)

        # frame dup or drop
        if aug:
            jitter_prob = tf.random_uniform([], minval=0, maxval=1)
            images, j = tf.case(
                {
                    tf.less(jitter_prob, JITTER_P / 2):
                        lambda: [images.write(j, image).write(j + 1, image), j + 2],  # duplicate frame
                    tf.greater(jitter_prob, JITTER_P):
                        lambda: [images.write(j, image), j + 1],  # normal write frame
                },
                default=lambda: [images, j],  # drop frame
                exclusive=True)
        else:
            images = images.write(i, image)
            j += 1
        return i + 1, j, images

    i, j, images = tf.while_loop(
        condition,
        loop_body,
        [i, j, images],
        back_prop=False,
        # parallel_iterations=VIDEO_LENGTH
    )
    x = images.stack()  # T x H x W x C

    if aug:
        x = tf.cond(flip_prob < FLIP_P, lambda: tf.reverse(x, axis=[2]),
                    lambda: x)  # flip
    return x


def cut_long(x, y):
    """if x lenth is longer than VIDEO_LENGTH, sample to VIDEO_LENGTH

    Args:
        x: 4-D tensor of shape (T, H, W, C).

    Returns: sampled x.

    """

    def sample_xy(x, y):
        T = tf.shape(x)[0]
        T = tf.cast(T, tf.float32)
        rate = T / VIDEO_LENGTH
        frame_ids = tf.range(0, T - 1, rate, dtype=tf.float32)
        frame_ids = tf.math.floor(frame_ids)
        frame_ids = tf.cast(frame_ids, tf.int32)
        return tf.cond(
            tf.shape(x)[0] <= VIDEO_LENGTH, lambda: x,
            lambda: tf.gather(x, frame_ids)), y

    def cut_xy(x, y):
        return x[:VIDEO_LENGTH, ...], tf.strings.substr(y, 0, LABEL_MAX_LENGTH)

    return tf.cond(
        tf.strings.length(y) <= LABEL_MAX_LENGTH, lambda: sample_xy(x, y),
        lambda: cut_xy(x, y))


def parse_single_example(serialized_record, aug, div_255):
    """parse serialized_record to tensors

    Args:
        serialized_record: One tfrecord example serialized.
        aug: Boolean. Add frame jitter and random flip if aug is True.

    Returns: TODO

    """
    features = tf.parse_single_example(serialized_record, EXAMPLE_FEATURES)
    # parse x
    video = features['video']
    video = tf.sparse_tensor_to_dense(video, default_value='')

    x = read_images(video, VIDEO_FRAME_SHAPE, 3, aug=aug)

    if div_255:
        x /= 255.0


    y0 = features['pinyin']
    y0 = tf.expand_dims(y0, 0)

    y1 = features['viseme']
    y1 = tf.sparse_tensor_to_dense(y1, default_value=0)


    y2 = features['word']
    y2 = tf.sparse_tensor_to_dense(y2, default_value=0)


    # return x,y
    inputs = {'video': x, 'unpadded_length': tf.shape(x)[0:1]}
    targets = {'pinyin': y0,'viseme': y1,'label': y2 }
    return (inputs, targets)



def tfrecord_input_fn(file_name_pattern,
                      mode=tf.estimator.ModeKeys.EVAL,
                      num_epochs=1,
                      batch_size=32,
                      aug=False,
                      div_255=True,
                      num_threads=1):
    """TODO: Docstring for grid_tfrecord_input_fn.

    Args:
        file_name_pattern: tfrecord filenames

    Kwargs:
        mode: train or others. Local shuffle will be performed if train.
        num_epochs: repeat data num_epochs times.
        batch_size: batch_size.
        aug: Boolean. Add frame jitter and random flip if aug is True.
        div_255: Boolean. The image pixel will be normalized to [0, 1] if div_255 is True.
        num_threads: Parallel thread number.

    Returns: TODO

    """
    file_names = tf.matching_files(file_name_pattern)
    dataset = tf.data.TFRecordDataset(
        filenames=file_names, num_parallel_reads=4)

    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100 * batch_size + 1)

    parse_func = partial(
        parse_single_example, maug=aug, div_255=div_255)
    dataset = dataset.map(parse_func, num_parallel_calls=num_threads)

    dataset = dataset.repeat(num_epochs)

    # dataset = dataset.batch(batch_size)
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=({
                           'video': [None, None, None, padded_channel],
                           'unpadded_length': [None]
                       }, {
                            'pinyin':[None],
                            'viseme': [None],
                            'label': [None]
                       }))
    dataset = dataset.prefetch(buffer_size=10)
    return dataset



