"""
ESRGAN Architecture

Input
Convolution filter (k9n64s1)
Residual Blocks (RRDB)- made up of three Dense Blocks (DB)
    Dense Blocks (DB)
        Convolution filter -> LeakyReLu (x3)
        Convolution filter
Upsampler by 2
Convolution filter (k3n64s1)
"""
import tensorflow as tf
import keras.layers as kl
from tensorflow._api.v2.nn import depth_to_space

class ESRGAN(object):

    def ConvBlock(self, xIn, featureMaps, leakyAlpha):
        xIn = kl.Conv2D(featureMaps, 9, 1, padding = "same")(xIn)
        xIn = kl.LeakyReLU(leakyAlpha)(self.genx)
        return xIn


    def UpsampleBlock(self, x, featureMaps, leakyAlpha):
        x = kl.Conv2D(featureMaps, 3, padding = "same")(x)
        x = depth_to_space(x, 2)
        x = kl.LeakyReLU(leakyAlpha)(x)
        return x


    def RRDBlock(self, xIn, featureMaps, leakyAlpha, residualScalar):
        ''' Residual in Residual Dense Block '''

        x = kl.Conv2D(featureMaps, 3, 1, padding = "same")(xIn)
        x1 = kl.LeakyReLU(leakyAlpha)(x)
        x1 = kl.Add() ([xIn, x1])

        x = kl.Conv2D(featureMaps, 3, 1, padding = "same")(x1)
        x2 = kl.LeakyReLU(leakyAlpha)(x)
        x2 = kl.Add() ([x1, x2])

        x = kl.Conv2D(featureMaps, 3, 1, padding = "same")(x2)
        x3 = kl.LeakyReLU(leakyAlpha)(x) 
        x3 = kl.Add() ([x2, x3])

        x = kl.Conv2D(featureMaps, 3, 1, padding = "same")(x3)
        x4 = kl.LeakyReLU(leakyAlpha)(x)
        x4 = kl.Add() ([x3, x4])

        x4 = kl.Conv2D(featureMaps, 3, 1, padding = "same")(x4)
        xSkip = kl.Add() ([self.genx, x4])

        ## scaling residual outputs with scalar between range(0,1)
        xSkip = kl.Lambda(lambda x : x * residualScalar) (xSkip)
        return xSkip
    
    def DiscriminatorBlock(self, x, featureMaps, strides, leakyAlpha):
        x = kl.Conv2D(featureMaps, 3, strides, padding = "same") (x)
        x = kl.BatchNormalization() (x)
        x = kl.LeakyReLU(leakyAlpha) (x)

        return x

    def Generator(self, scalingFactor, featureMaps, residualBlocks, leakyAlpha, residualScalar):
        '''
        scalingFactor (double)  : determining factor for output image scaling
        featureMaps (int)       : number of convBlocks
        residualBlocks (int)    : number of residual blocks
        leakyAlpha (double)     : factor determining threshold value of LeakyReLu
        residualScalar (double) : value that keep output of residual blocks scaled - to stabilize training

        '''
        
        ## initalize input layer
        ## scaling pixels to in range(0, 1)
        input = tf.keras.Input((None, None, 3))
        xIn = kl.Rescaling(scale = 1.0/255, offset = 0.0)(input)

        ## Convolution Filter Block
        xIn = self.ConvBlock(xIn, featureMaps, leakyAlpha)

        ## Residual in Residual Blocks
        for block in range(residualBlocks) :
            xSkip = self.RRDBlock(xIn, featureMaps, leakyAlpha, residualScalar)
        
        ## 
        x = kl.Conv2D(featureMaps, 3, padding = "same")(xSkip)
        x = kl.Add() ([xIn, x])

        ## UpSample
        x = self.UpsampleBlock(x, featureMaps * (scalingFactor // 2), leakyAlpha)
        x = self.UpsampleBlock(x, featureMaps, leakyAlpha)

        ## Output layer
        x = kl.Conv2D(3, 9, 1, padding = "same", activation = "tanh") (x)
        output = kl.Rescaling(scale = 127.5, offset = 127.5) (x)

        ## Generator Model
        generator = tf.keras.Model(input, output)

        return generator

    def Discriminator(self, featureMaps, leakyAlpha, discBlocks):

        ## Input Layers
        input = tf.keras.Input((None, None, 3))
        x = kl.Rescaling(scale = 1.0/127.5, offset = -1) (input)
        x = self.ConvBlock(x, featureMaps, leakyAlpha)

        x = self.DiscriminatorBlock(x, featureMaps, 1, leakyAlpha)

        for i in range(1, discBlocks):
            x = self.DiscriminatorBlock(x, featureMaps * (2**i), 2, leakyAlpha)
            x = self.DiscriminatorBlock(x, featureMaps * (2**i), 1, leakyAlpha)

        x = kl.GlobalAveragePooling2D() (x)
        x = kl.LeakyReLU(leakyAlpha) (x)

        x = kl.Dense(1, activation = "sigmoid") (x)

        discriminator = tf.keras.Model(input, x)

        return discriminator
    