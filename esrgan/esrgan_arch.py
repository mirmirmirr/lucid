import cv2
import os
import random
import opendatasets as od

# tensorflow dependencies (model compenents and deep learning components)
# using tensorflow funcational api
import tensorflow as tf
from tensorflow.keras import layers as kl
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class ESRGAN(object):
    def RDB(self, x):
        x1 = kl.Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x)
        x1 = kl.concatenate([x, x1])

        x2 = kl.Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x1)
        x2 = kl.concatenate([x, x1, x2])

        x3 = kl.Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x2)
        x3 = kl.concatenate([x, x1, x2, x3])

        x4 = kl.Conv2D(32, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(x3)
        x4 = kl.concatenate([x, x1, x2, x3, x4])

        x5 = kl.Conv2D(64, 3, padding="same")(x4)
        x5 = kl.Lambda(lambda x : x * 0.2)(x5)
        xSkip = kl.Add()([x, x5])

        return xSkip

    def RRDB(self, x_input):
        x = self.RDB(x_input)
        x = self.RDB(x)
        x = self.RDB(x)
        x = kl.Lambda(lambda x : x * 0.2)(x)
        x = kl.Add()([x_input, x])
        return x

    def generator(self):
        input = kl.Input(shape=(None, None, 3), name='input_image')
        xIn = kl.Rescaling(scale=1.0/255, offset=0.0)(input)

        x = kl.Conv2D(64, 3, padding="same", activation=kl.LeakyReLU(alpha=0.2))(xIn)
        x1 = self.RRDB(x)

        for block in range(15):
            x1 = self.RRDB(x1)

        xSkip = kl.Conv2D(64, 3, padding="same")(x1)
        xSkip = kl.concatenate([x, xSkip])

        x = kl.Conv2D(128, 3, padding="same")(xSkip)
        x = tf.nn.depth_to_space(x, 2)
        x = kl.LeakyReLU(alpha=0.2)(x)

        x = kl.Conv2D(64, 3, padding="same")(x)
        x = tf.nn.depth_to_space(x, 2)
        x = kl.LeakyReLU(alpha=0.2)(x)

        x = kl.Conv2D(3, 3, padding="same", activation="tanh")(x)
        output = kl.Rescaling(scale=127.5, offset=127.5)(x)

        return Model(input, output, name='generator')

    def discriminator(self, featureMaps, leakyAlpha, discBlocks):
        inputs = kl.Input((None, None, 3))
        x = kl.Rescaling(scale=1.0/127.5, offset=-1)(inputs)
        x = kl.Conv2D(featureMaps, 3, padding="same")(x)
        x = kl.LeakyReLU(leakyAlpha)(x)

        x = kl.Conv2D(featureMaps, 3, padding="same")(x)
        x = kl.BatchNormalization()(x)
        x = kl.LeakyReLU(leakyAlpha)(x)

        downConvConf = {
            "strides": 2,
            "padding": "same",
        }

        for i in range(1, discBlocks):
            x = kl.Conv2D(featureMaps * (2 ** i), 3, **downConvConf)(x)
            x = kl.BatchNormalization()(x)
            x = kl.LeakyReLU(leakyAlpha)(x)

            x = kl.Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = kl.BatchNormalization()(x)
            x = kl.LeakyReLU(leakyAlpha)(x)

        x = kl.GlobalAvgPool2D()(x)
        x = kl.LeakyReLU(leakyAlpha)(x)
        x = kl.Dense(1, activation="sigmoid")(x)

        return Model(inputs, x)