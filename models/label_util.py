#!/usr/bin/env python
# -*- coding: utf-8 -*-


import string
import tensorflow as tf

from transformer_official.utils.tokenizer import PAD, PAD_ID, EOS, EOS_ID, RESERVED_TOKENS




def label_to_numeric(labels, dic):
    """ preprocess labels to satisfy the need of model.

    Args:
        labels: 1-D Tensor. labels of shape (batch_size,). For example: [ 'ab haha', 'hha fd fd']

    Returns: 2-D int32 Tensor with shape  [batch_size, T]  . labels with numeric shape. EOS is padded to each label.

    """
    pad_tensor = tf.constant([EOS], tf.string)
    # append 'E' to label
    eos = tf.tile(pad_tensor, tf.shape(labels))  # Bx1
    labels = tf.string_join([labels, eos])

    label_char_list = string2char_list(labels)  # B x T

    numeric_label = string2indices(
        label_char_list, dic=dic)  # SparseTensor, dense_shape: [B x T]
    numeric_label = tf.sparse_tensor_to_dense(
        numeric_label, default_value=PAD_ID)  #[B x T]

    # paddings = tf.concat(
    # [[0, 0, 0],
    # self.params['target_max_len'] - tf.shape(numeric_label)[1:2]], 0)
    # paddings = tf.reshape(paddings, [2, 2])  # [[0,0], [0, pad_n]]
    # numeric_label = tf.pad(numeric_label, paddings)
    # numeric_label.set_shape(
    # [self.params['batch_size'],
    # self.params['target_max_len']])  # target_max_len
    return numeric_label


def indices2string(predictions, dic):
    """map numeric int64 indices to char values.
    For example:
        [ 0, 1, 2] -> [ 'a', 'b', 'c']

    Args:
        predictions: Tensor or SparseTenser. int64 indices to convert

    Returns: The same type of input. map the corresponding values in the DICT.
    """
    if predictions.dtype != tf.int64:
        predictions = tf.cast(predictions, tf.int64)
    index2string_table = tf.contrib.lookup.index_to_string_table_from_tensor(
        dic, default_value='_')
    return index2string_table.lookup(predictions)


def string2indices(s, dic):
    """ map the char values in `s` to numeric int64 indices
    For example:
        [ 'a', 'b', 'c'] -> [ 0, 1, 2]

    Args:
        s: Tensor or SparseTensor.

    Returns: The same type of input. the Dtype is `tf.int64`

    """
    string2index_table = tf.contrib.lookup.index_table_from_tensor(
        dic, num_oov_buckets=1, default_value=-1)
    return string2index_table.lookup(s)


def string_join(string_tensor):
    """TODO: Docstring for string_join.

    Args:
        string_tensor: 1-D or 2-D string Tensor. join the strings in the last dimension

    Returns: 0-D or 1-D string Tensor. The joined string.

    """
    i = tf.constant(0)
    string_num = tf.shape(string_tensor)[-1]
    if tf.rank(string_tensor) == 1:
        joined_string = ''
    else:
        joined_string = tf.tile([''],
                                tf.expand_dims(tf.shape(string_tensor)[0], 0))

    c = lambda i, joined_string: tf.less(i, string_num)
    b = lambda i, joined_string: [tf.add(i,1), joined_string+string_tensor[...,i]]
    i, joined_string = tf.while_loop(c, b, [i, joined_string], back_prop=False)
    return joined_string


def char_list2string(char_list):
    """convert a char based list to string. For example:
    [ 'a', 'b'] -> [ 'ab']

    Args:
        char_list: Tenosr or SparseTensor of rank 2.

    Returns: Tensor of rank 1, dtype `string`.

    """
    #char_list: SparseTensor: [ ['a', 'b'], ['a','b','c'] ]
    if isinstance(char_list, tf.SparseTensor):
        dense_list = tf.sparse_tensor_to_dense(
            char_list, '')  # Tensor: [[ 'a' , 'b', ''], [ 'a', 'b', 'c'] ]
    else:
        dense_list = char_list

    return string_join(dense_list)


def string2char_list(string):
    """convert a string to char based list. For example:
    [ 'ab'] -> [ 'a', 'b']

    Args:
        string: `1-D` string `Tensor`. The string to convert

    Returns: `2-D` string `SparseTensor`. The string are split to list of chars.
    """
    return tf.string_split(string, delimiter='')


def char2word(predicted_char_list):
    """ convert char based string to word based string. Suppose the words are divided with space ' '.
    For example: [ ['a', 'b', ' ', 'd']] convert to [['ab', 'd'] ]

    Args:
        predicted_char_list: SparseTensor. dense_shape (batch_size, string_char_len)

    Returns: SparseTensor. dense_shape (batch_size, string_word_len)

    """
    char_string = char_list2string(predicted_char_list)  # [['ab d']]
    word_list = tf.string_split(char_string)  # [[ 'ab', 'd']]
    return word_list
