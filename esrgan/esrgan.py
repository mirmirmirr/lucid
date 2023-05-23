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

import keras.layers as kl
from tensorflow._api.v2.nn import depth_to_space

class ESRGAN(object):

    def ConvBlock(self, xIn, featureMaps, leakyAlpha):
        xIn = kl.Conv2D(featureMaps, 9, 1, padding = "same")(xIn)
        xIn = kl.LeakyReLU(leakyAlpha)(self.genx)
        return xIn


    def UpsampleBlock():
        pass

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
        input = kl.Input((None, None, 3))
        xIn = kl.Rescaling(scale = 1.0/255, offset = 0.0)(input)

        ## Convolution Filter Block
        xIn = self.ConvBlock(xIn, featureMaps, leakyAlpha)

        ## Residual in Residual Blocks
        for block in range(residualBlocks) :
            xSkip = self.RRDBlock(xIn, featureMaps, leakyAlpha, residualScalar)


    def Discriminator():
        pass
    