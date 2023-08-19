import cv2
import os
import random
import opendatasets as od

# tensorflow dependencies (model compenents and deep learning components)
# using tensorflow funcational api
import tensorflow as tf
from keras.models import Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow_datasets as tfds

import keras.layers as kl
from tensorflow._api.v2.nn import depth_to_space
import numpy as np

class Training(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()
        self.gen = generator
        self.disc = discriminator
        self.vgg = vgg
        self.batchSize = batchSize

        self.gen_opt = tf.keras.optimizers.Adam(3e-5)
        self.disc_opt = tf.keras.optimizers.Adam(3e-5)
        self.bceLoss = tf.losses.BinaryCrossentropy()
        self.mseLoss = tf.losses.MeanSquaredError()

    # def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
    #     super().compile()
	# 	# initialize the optimizers for the generator
	# 	# and discriminator
    #     self.gen_opt = gOptimizer
    #     self.disc_opt = dOptimizer

	# 	# initialize the loss functions
    #     self.bceLoss = bceLoss
    #     self.mseLoss = mseLoss

    def train_step(self, images):
        (loRes, hiRes) = images
        loRes = tf.cast(loRes, tf.float32)
        hiRes = tf.cast(hiRes, tf.float32)

        superImages = self.gen(loRes)
        combined = tf.concat([superImages, hiRes], axis=0)

        labels = tf.concat([tf.zeros((self.batchSize, 1)), tf.ones((self.batchSize, 1))], axis=0)

        ## DISCRIMINATOR TRAINING
        with tf.GradientTape() as tape:
            rawPred = self.disc(combined)
            rawFake = rawPred[:self.batchSize]
            rawReal = rawPred[self.batchSize:]

            predFake = tf.keras.activations.sigmoid(rawFake - tf.math.reduce_mean(rawReal))
            predReal = tf.keras.activations.sigmoid(rawReal - tf.math.reduce_mean(rawFake))

            predictions = tf.concat([predFake, predReal], axis=0)
            disc_loss = self.bceLoss(labels, predictions)

        # calculate gradients
        grad = tape.gradient(disc_loss, self.disc.trainable_variables)

        # calculate weights and apply to model
        self.disc_opt.apply_gradients(zip(grad, self.disc.trainable_variables))

        # generate misleading labels for generator training
        misleaders = tf.ones((self.batchSize, 1))

        ## GENERATOR TRAINING
        with tf.GradientTape() as tape:
            imposterImages = self.gen(loRes)

            rawPred = self.disc(imposterImages)
            realPred = self.disc(hiRes)
            predictions = tf.keras.activations.sigmoid(rawPred - tf.math.reduce_mean(realPred))

            gen_loss = self.bceLoss(misleaders, predictions)
            pixel_loss = self.mseLoss(hiRes, imposterImages)

            superVGG = tf.keras.applications.vgg19.preprocess_input(imposterImages)
            superVGG = self.vgg(superVGG) / 12.75
            hiresVGG = tf.keras.applications.vgg19.preprocess_input(hiRes)
            hiresVGG = self.vgg(hiresVGG) / 12.75

            perceptualLoss = self.mseLoss(hiresVGG, superVGG)
            totalLoss = 5e-3 * gen_loss + perceptualLoss + 1e-2 * pixel_loss

        # calc gradients
        grad = tape.gradient(totalLoss, self.gen.trainable_variables)

        # optimize weights
        self.gen_opt.apply_gradients(zip(grad, self.gen.trainable_variables))

        return {"disc": disc_loss, 'gen_totalLoss': totalLoss, "gen_loss": gen_loss, "perceptual_loss": perceptualLoss, "pixel_loss": pixel_loss}

# def train(data, pretrainEPOCHS, finetuneEPOCHS):
#     # loss = Losses()

#     gen.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=mseLoss)

#     print("pretraining ESRGAN generator...")
#     gen.fit(data, epochs=pretrainEPOCHS, steps_per_epoch=10, batch_size=64)

#     if not os.path.exists('outputs'):
#       os.makedirs('outputs')

#     print('saving generator...')
#     gen.save(PRETRAINED_GEN_MODEL)

#     vgg = VGG.build()
#     esrgan = Training(gen, disc, vgg, 64)

#     print("training ESRGAN...")
#     esrgan.fit(data, epochs=finetuneEPOCHS, steps_per_epoch=10)

#     print('saving ESRGAN...')
#     esrgan.gen.save(GENERATOR_MODEL)

#     print('SAVED ALL MODELS ARE SAVE FINALLY AHHHHHHHHHHHHHHHHHHHHHHH')

if __name__ == "__main__":
    DATASET = "div2k/bicubic_x4"

    DIV2K_PATH = os.path.join('dataset', "div2k")
    GPU_TRAIN_PATH = os.path.join('tfrecord', 'train')
    GPU_TEST_PATH = os.path.join('tfrecord', 'test')

    # path to our base output directory
    BASE_OUTPUT_PATH = "outputs"

    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for GPU training
    tfrTrainPath = GPU_TRAIN_PATH

    # grab train TFRecord filenames
    print("[INFO] grabbing the train TFRecords...")
    trainTfr = tf.io.gfile.glob(tfrTrainPath +"/*.tfrec")

    # build the div2k datasets from the TFRecords
    print("[INFO] creating train and test dataset...")
    trainDs = justload(filenames=trainTfr, train=True,batchSize=64)

    PRETRAINED_GEN_MODEL = os.path.join('outputs', 'models', 'pretrained_generator')
    GENERATOR_MODEL = os.path.join('outputs', 'models', 'generator')
    pretrainEPOCHS = 1500
    finetuneEPOCHS = 1000

    # train(trainDs, 1500, 1000)
    gen.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=mseLoss)

    print("pretraining ESRGAN generator...")
    gen.fit(trainDs, epochs=pretrainEPOCHS, steps_per_epoch=10, batch_size=64)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    print('saving generator...')
    gen.save(PRETRAINED_GEN_MODEL)

    vgg = VGG.build()
    esrgan = Training(gen, disc, vgg, 64)

    print("training ESRGAN...")
    esrgan.fit(trainDs, epochs=finetuneEPOCHS, steps_per_epoch=10)

    print('saving ESRGAN...')
    esrgan.gen.save(GENERATOR_MODEL)

    print('SAVED ALL MODELS ARE SAVE FINALLY AHHHHHHHHHHHHHHHHHHHHHHH')