import cv2
import os
import random
import opendatasets as od

# tensorflow dependencies (model compenents and deep learning components)
# using tensorflow funcational api
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dense, Input, concatenate, Lambda, Add
import tensorflow_datasets as tfds

import keras.layers as kl
import numpy as np

class ESRGAN(object):
    def RDB(self, x):
        '''
        Residual Dense Block
        '''
        x1 = Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x)
        x1 = concatenate([x, x1])

        x2 = Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x1)
        x2 = concatenate([x, x1, x2])

        x3 = Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x2)
        x3 = concatenate([x, x1, x2, x3])

        x4 = Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x3)
        x4 = concatenate([x, x1, x2, x3, x4])

        x5 = Conv2D(64, 3, padding="same")(x4)
        x5 = Lambda(lambda x : x * 0.2) (x5)
        xSkip = Add() ([x, x5])

        return xSkip

    def RRDB(self, x_input):
        '''
        Residual in Residual Dense Block
        '''
        x = self.RDB(x_input)
        x = self.RDB(x)
        x = self.RDB(x)
        x = Lambda(lambda x : x * 0.2) (x)
        x = Add() ([x_input, x])
        return x

    def generator(self):
        input = Input(shape=(None, None, 3), name='input_image')
        xIn = kl.Rescaling(scale=1.0/255, offset=0.0)(input)

        # conv block with leaky activation
        x = Conv2D(64, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(xIn)

        x1 = self.RRDB(x)

        # residual in residual blocks
        for block in range(15):
            x1 = self.RRDB(x1)

        xSkip = Conv2D(64, 3, padding="same")(x1)
        xSkip = concatenate([x, xSkip])

        # upscaling
        x = Conv2D(128, 3, padding="same")(xSkip)
        x = tf.nn.depth_to_space(x, 2)
        x = kl.LeakyReLU(alpha=0.2) (x)

        x = Conv2D(64, 3, padding="same")(x)
        x = tf.nn.depth_to_space(x, 2)
        x = kl.LeakyReLU(alpha=0.2) (x)

        # back to conv blocks
        x = Conv2D(3, 3, padding="same", activation="tanh") (x)
        output = kl.Rescaling(scale=127.5, offset=127.5) (x)

        return Model(input, output, name='generator')

    def disc_block(self, x, featureMaps):
        x = Conv2D(featureMaps, 3, strides=1, padding='same')(x)
        x = kl.BatchNormalization() (x)
        x = kl.LeakyReLU(alpha=0.2) (x)

        x = Conv2D(featureMaps, 3, strides=2, padding='same')(x)
        x = kl.BatchNormalization() (x)
        x = kl.LeakyReLU(alpha=0.2) (x)

        return x

    def discriminator(self):
        input = Input(shape=(None, None, 3), name='input_image')
        x = kl.Rescaling(scale=1.0/127.5, offset=-1)(input)
        x = Conv2D(64, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x)

        x = Conv2D(64, 3, strides=2, padding='same')(x)
        x = kl.BatchNormalization() (x)
        x = kl.LeakyReLU(alpha=0.2) (x)

        for i in range(1, 4):
            x = self.disc_block(x, 64 * (2 ** i))

        # Dense block (1024)
        x = kl.GlobalAveragePooling2D() (x)
        x = kl.LeakyReLU(alpha=0.2) (x)

        # Dense (1)
        x = Dense(1, activation='sigmoid') (x)

        return Model(input, x, name='discriminator')