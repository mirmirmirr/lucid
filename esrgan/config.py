import os
import albumentations as a

CHECKPOINT_GEN = "RealESRGAN_x4plus.pth"
DEVICE = "cpu"

## training specifications
PRETRAIN_LR = 1e-4
FINETUNE_LR = 3e-5

PRETRAIN_EPOCHS = 1500
FINETUNE_EPOCHS = 1000
STEPS_PER_EPOCH = 10

## ESRGAN model specifications
FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 16
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4
RESIDUAL_SCALAR = 0.2