#!/usr/bin/env python
# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   file name: cnn_extractor.py
#   author: klaus
#   email: klaus.cheng@qq.com
#   created date: 2018/07/10
#   description:
#
#================================================================

import tensorflow as tf
import tensorflow.contrib.keras as keras


class CNN(object):
    """base cnn model. Extract feature of the video_tensor.

    Input:
        video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.
    Output: Tensor of shape(T, feature_len)

    """

    def __init__(self, feature_len, training, scope='cnn_feature_extractor'):
        self.feature_len = feature_len
        self.training = training
        self.scope = scope

    def build():
        raise NotImplementedError('CNN not NotImplemented.')


class EarlyFusion2D(CNN):
    """early fusion + 2D cnn"""

    def __init__(self, *args, **kwargs):
        super(EarlyFusion2D, self).__init__(*args, **kwargs)

    def build(self, video_tensor):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope):
            self.conv1 = keras.layers.Conv3D(
                32, (5, 5, 5),
                strides=(1, 2, 2),
                padding='same',
                kernel_initializer='he_normal',
                name='conv1')(video_tensor)
            self.batc1 = tf.layers.batch_normalization(
                self.conv1, training=self.training, name='batc1')
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxp1 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max1')(self.drop1)

            self.conv2 = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    64, (5, 5),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal'),
                name="TD_conv2")(self.maxp1)
            self.batc2 = tf.layers.batch_normalization(
                self.conv2, training=self.training, name='batc2')
            self.actv2 = keras.layers.Activation(
                'relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0.5)(self.actv2)
            self.maxp2 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max2')(self.drop2)

            self.conv3 = keras.layers.TimeDistributed(
                keras.layers.Conv2D(
                    96, (3, 3),
                    strides=(1, 1),
                    padding='same',
                    kernel_initializer='he_normal'),
                name="TD_conv3")(self.maxp2)
            self.batc3 = tf.layers.batch_normalization(
                self.conv3, training=self.training, name='batc3')
            self.actv3 = keras.layers.Activation(
                'relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0.5)(self.actv3)
            self.maxp3 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max3')(self.drop3)

            # prepare output
            self.conv4 = keras.layers.Conv3D(
                self.feature_len, (1, 1, 1),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv4')(self.maxp3)
            self.output = keras.layers.TimeDistributed(
                # keras.layers.GlobalAveragePooling2D(name='global_ave1'),
                keras.layers.GlobalMaxPool2D(name='global_ave1'),
                name='TD_GMP1')(self.conv4)  #shape: (T, feature_len)
            return self.output


class LipNet(CNN):
    """lipnet cnn feature extractor"""

    def __init__(self, *args, **kwargs):
        super(LipNet, self).__init__(*args, **kwargs)

    def build(self, video_tensor):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). In GRID. H = 50 and W = 100.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            print("***************input_shape:", video_tensor.get_shape())
            self.zero1 = keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero1')(video_tensor)
            self.conv1 = keras.layers.Conv3D(
                32, (3, 5, 5),
                strides=(1, 2, 2),
                kernel_initializer='he_normal',
                name='conv1')(self.zero1)
            # self.batc1 = keras.layers.BatchNormalization(name='batc1')(
            # self.conv1, training=self.training)
            self.batc1 = tf.layers.batch_normalization(
                self.conv1, training=self.training, name='batc1')
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxp1 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max1')(self.drop1)

            print("***************conv1_shape:", self.maxp1.get_shape())

            self.zero2 = keras.layers.ZeroPadding3D(
                padding=(1, 2, 2), name='zero2')(self.maxp1)
            self.conv2 = keras.layers.Conv3D(
                64, (3, 5, 5),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv2')(self.zero2)
            # self.batc2 = keras.layers.BatchNormalization(name='batc2')(
            # self.conv2, training=self.training)
            self.batc2 = tf.layers.batch_normalization(
                self.conv2, training=self.training, name='batc2')
            self.actv2 = keras.layers.Activation(
                'relu', name='actv2')(self.batc2)
            self.drop2 = keras.layers.SpatialDropout3D(0.5)(self.actv2)
            self.maxp2 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max2')(self.drop2)

            self.zero3 = keras.layers.ZeroPadding3D(
                padding=(1, 1, 1), name='zero3')(self.maxp2)
            self.conv3 = keras.layers.Conv3D(
                96, (3, 3, 3),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv3')(self.zero3)
            # self.batc3 = keras.layers.BatchNormalization(name='batc3')(
            # self.conv3, training=self.training)
            self.batc3 = tf.layers.batch_normalization(
                self.conv3, training=self.training, name='batc3')
            self.actv3 = keras.layers.Activation(
                'relu', name='actv3')(self.batc3)
            self.drop3 = keras.layers.SpatialDropout3D(0.5)(self.actv3)
            self.maxp3 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max3')(self.drop3)

            # prepare output
            self.conv4 = keras.layers.Conv3D(
                self.feature_len, (1, 1, 1),
                strides=(1, 1, 1),
                kernel_initializer='he_normal',
                name='conv4')(self.maxp3)
            self.output = keras.layers.TimeDistributed(
                keras.layers.GlobalMaxPooling2D(name='global_ave1'),
                name='timeDistributed1')(self.conv4)  #shape: (T, feature_len)
        print("output_shape:", self.output.get_shape())
        return self.output

class ResNet(CNN):
    """lipnet cnn feature extractor"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def build(self, video_tensor):
        """build model.

        Args:
            video_tensor: Tensor. videos of shape (batch_size, T, H, W, C). H = 64 and W = 128.

        Returns: the output tensor of the model

        """
        with tf.variable_scope(self.scope):

            self.conv1 = keras.layers.Conv3D(
                64, (5, 5, 5),
                strides=(1, 2, 2),
                kernel_initializer='he_normal',
                name='conv1')(video_tensor)
            self.batc1 = tf.layers.batch_normalization(
                self.conv1, training=self.training, name='batc1')
            self.actv1 = keras.layers.Activation(
                'relu', name='actv1')(self.batc1)
            self.drop1 = keras.layers.SpatialDropout3D(0.5)(self.actv1)
            self.maxp1 = keras.layers.MaxPooling3D(
                pool_size=(1, 2, 2), strides=(1, 2, 2),
                name='max1')(self.drop1)

            self.res1 = self.res_block(filters=64, kernel_size=(3, 3), strides=(1, 1), is_first_block_of_first_layer=True)(self.maxp1)
            self.res2 = self.res_block(filters=64, kernel_size=(3, 3), strides=(1, 1))(self.res1)
            self.res3 = self.res_block(filters=128, kernel_size=(3, 3), strides=(2, 2))(self.res2)
            self.res4 = self.res_block(filters=128, kernel_size=(3, 3), strides=(1, 1))(self.res3)
            self.res5 = self.res_block(filters=256, kernel_size=(3, 3), strides=(2, 2))(self.res4)
            self.res6 = self.res_block(filters=256, kernel_size=(3, 3), strides=(1, 1))(self.res5)
            self.res7 = self.res_block(filters=self.feature_len, kernel_size=(3, 3), strides=(2, 2))(self.res6)
            self.res8 = self.res_block(filters=self.feature_len, kernel_size=(3, 3), strides=(1, 1))(self.res7)


            self.output = keras.layers.TimeDistributed(
                keras.layers.GlobalMaxPooling2D(name='global_ave1'),
                name='timeDistributed1')(self.res8)  #shape: (T, feature_len)
            return self.output

    def res_block(self, filters, kernel_size=(3, 3), strides=(1, 1), is_first_block_of_first_layer=False):
            def f(input):


                shortcut = keras.layers.TimeDistributed(
                              keras.layers.Conv2D(
                              filters=filters,
                              kernel_size=(1, 1),
                              strides=strides,
                              kernel_initializer='he_normal',
                              padding='same'))(input)

                if is_first_block_of_first_layer:
                    # don't repeat bn->relu since we just did bn->relu->maxpool
                    conv = keras.layers.TimeDistributed(
                              keras.layers.Conv2D(
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              kernel_initializer='he_normal',
                              padding='same'))(input)

                else:
                    conv = self._bn_relu_conv2d(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=strides,
                                            )(input)

                residual = self._bn_relu_conv2d(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=(1, 1)
                                                )(conv)

                return shortcut + residual

            return f



    def _bn_relu_conv2d(self, **conv_params):
            """Helper to build a  BN -> relu -> conv2d block."""
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params["strides"]

            def f(input):
                normalization = tf.layers.batch_normalization(
                input, training=self.training)
                activation = keras.layers.Activation(
                'relu')(normalization)
                return keras.layers.TimeDistributed(
                              keras.layers.Conv2D(
                              filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              kernel_initializer='he_normal',
                              padding='same'))(activation)

            return f


