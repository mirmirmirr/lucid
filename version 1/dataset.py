import albumentations as a
import tensorflow as tf

'''
(applies to both LOWRES and HIGHRES images)
random crop 
get center crop
random flip
random rotate

'''

def read_training(example):
    
    ## getting a feature template
    feature = {
        "lowRes" : tf.io.FixedLenFeature(shape=[], dtype= tf.string),
        "highRes" : tf.io.FixedLenFeature(shape = [], dtype = tf.string)
    }

    ## using parse_single_example 
    # (parses an instance of the data and returns a dictionary with "lowRes" and "highRes" as keys and the images as values)
    example = tf.io.parse_single_example(example, feature)

    ## getting the lowRes and highRes images
    lowRes = tf.io.parse_tensor(example["lowRes"], out_type = tf.uint8)
    highRes = tf.io.parse_tensor(example["highRes"], out_type = tf.uint8)

    transform = a.Compose([
        a.RandomCrop(width = 128, height = 128),
        a.HorizontalFlip(),
        a.RandomRotate90()
    ])

    lr_transform = a.Compose([
        
    ])
