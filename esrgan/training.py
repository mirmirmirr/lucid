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

from loss import VGG, Losses
from data.preperation import justload
from esrgan_arch import ESRGAN

class Training(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()
        self.gen = generator
        self.disc = discriminator
        self.vgg = vgg
        self.batchSize = batchSize

    def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
        super().compile()
        self.gen_opt = gOptimizer
        self.disc_opt = dOptimizer
        self.bceLoss = bceLoss
        self.mseLoss = mseLoss

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
            rawFake, rawReal = tf.split(rawPred, num_or_size_splits=2, axis=0)

            predFake = tf.keras.activations.sigmoid(rawFake - tf.reduce_mean(rawReal))
            predReal = tf.keras.activations.sigmoid(rawReal - tf.reduce_mean(rawFake))

            predictions = tf.concat([predFake, predReal], axis=0)
            dLoss = self.bceLoss(labels, predictions)

        grad = tape.gradient(dLoss, self.disc.trainable_variables)
        self.disc_opt.apply_gradients(zip(grad, self.disc.trainable_variables))

        misleaders = tf.ones((self.batchSize, 1))

        ## GENERATOR TRAINING
        with tf.GradientTape() as tape:
            imposterImages = self.gen(loRes)
            rawPred = self.disc(imposterImages)
            realPred = self.disc(hiRes)
            predictions = tf.keras.activations.sigmoid(rawPred - tf.reduce_mean(realPred))

            gen_loss = self.bceLoss(misleaders, predictions)
            pixel_loss = self.mseLoss(hiRes, imposterImages)

            superVGG = tf.keras.applications.vgg19.preprocess_input(imposterImages)
            superVGG = self.vgg(superVGG)

            realVGG = tf.keras.applications.vgg19.preprocess_input(hiRes)
            realVGG = self.vgg(realVGG)

            perc_loss = self.mseLoss(realVGG, superVGG)
            gLoss = pixel_loss + 0.001 * gen_loss + 0.006 * perc_loss

        grad = tape.gradient(gLoss, self.gen.trainable_variables)
        self.gen_opt.apply_gradients(zip(grad, self.gen.trainable_variables))

        return {"dLoss": dLoss, "gLoss": gLoss}

if __name__ == "__main__":

    ## initalizing opimizers and loss functions
    gen_opt_pt = tf.keras.optimizers.legacy.Adam(1e-4)
    disc_opt_pt = tf.keras.optimizers.legacy.Adam(1e-4)

    gen_opt = tf.keras.optimizers.legacy.Adam(3e-5)
    disc_opt = tf.keras.optimizers.legacy.Adam(3e-5)

    bceLoss = tf.losses.BinaryCrossentropy()
    mseLoss = tf.losses.MeanSquaredError()

    PRETRAINED_GEN_MODEL = os.path.join('outputs', 'models', 'pretrained_generator')
    GENERATOR_MODEL = os.path.join('outputs', 'models', 'generator')
    pretrainEPOCHS = 1500
    finetuneEPOCHS = 1000

    DATASET = "div2k/bicubic_x4"

    DIV2K_PATH = os.path.join('dataset', "div2k")
    GPU_TRAIN_PATH = os.path.join('tfrecord', 'train')
    GPU_TEST_PATH = os.path.join('tfrecord', 'test')

    # path to our base output directory
    BASE_OUTPUT_PATH = "outputs"

    ## define multi-gpu strategy
    strategy = tf.distribute.MirroredStrategy()

    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for GPU training
    tfrTrainPath = GPU_TRAIN_PATH
    pretrainedGenPath = PRETRAINED_GEN_MODEL
    genPath = GENERATOR_MODEL

    # set the batch size
    BATCH_SIZE = 64

    # grab train TFRecord filenames
    print("[INFO] grabbing the train TFRecords...")
    trainTfr = tf.io.gfile.glob(tfrTrainPath +"/*.tfrec")

    # build the div2k datasets from the TFRecords
    print("[INFO] creating train and test dataset...")
    trainDs = justload(filenames=trainTfr, train=True, batchSize=BATCH_SIZE * strategy.num_replicas_in_sync)
    # trainDs = load_dataset(filenames=trainTfr, train=True,
    # 	batchSize=64)

    ESRGAN = ESRGAN()
    gen = ESRGAN.generator()
    disc = ESRGAN.discriminator(4, 0.2, 4)

    # train(trainDs, 1500, 1000)
    with strategy.scope():
        losses = Losses(numReplicas = strategy.num_replicas_in_sync)
        ESRGAN = ESRGAN()
        gen = ESRGAN.generator()
        gen.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=losses.mse_loss)

        print("pretraining ESRGAN generator...")
        # gen.fit(trainDs, epochs=pretrainEPOCHS, steps_per_epoch=10, batch_size=64)
        gen.fit(trainDs, epochs=pretrainEPOCHS, steps_per_epoch=10)

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    print('saving generator...')
    gen.save(pretrainedGenPath)

    with strategy.scope():
        losses = Losses(numReplicas = strategy.num_replicas_in_sync)

        ESRGAN = ESRGAN()
        gen = ESRGAN.generator()
        disc = ESRGAN.discriminator(4, 0.2, 4)

        vgg = VGG.build()
        esrgan = Training(gen, disc, vgg, batchSize=64)
        esrgan.compile(tf.keras.optimizers.Adam(3e-5), tf.keras.optimizers.Adam(3e-5), losses.bce_loss, losses.mse_loss)

        print("training ESRGAN...")
        esrgan.fit(trainDs, epochs=finetuneEPOCHS, steps_per_epoch=10)

    print('saving ESRGAN...')
    esrgan.gen.save(genPath)

    print('SAVED ALL MODELS ARE SAVE FINALLY AHHHHHHHHHHHHHHHHHHHHHHH')