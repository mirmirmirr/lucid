import cv2
import os
import random
import opendatasets as od

# tensorflow dependencies (model compenents and deep learning components)
# using tensorflow funcational api
import tensorflow as tf
from keras.models import Model
import tensorflow_datasets as tfds

import keras.layers as kl
import numpy as np

## VGG model for perception loss
class VGG:
    @staticmethod
    def build():
        vgg = tf.keras.applications.VGG19(input_shape=(None, None, 3), weights='imagenet', include_top=False)
        return Model(vgg.input, vgg.layers[20].output)

## loss calculations
class Losses:
	def __init__(self, numReplicas):
		self.numReplicas = numReplicas
	def bce_loss(self, real, pred):
		# compute binary cross entropy loss without reduction
		BCE = tf.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
		loss = BCE(real, pred)
		# compute reduced mean over the entire batch
		loss = tf.reduce_mean(loss) * (1. / self.numReplicas)
		# return reduced bce loss
		return
	def mse_loss(self, real, pred):
		# compute mean squared error loss without reduction
		MSE = tf.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
		loss = MSE(real, pred)
		# compute reduced mean over the entire batch
		loss = tf.reduce_mean(loss) * (1. / self.numReplicas)
		# return reduced mse loss
		return loss