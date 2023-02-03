# -*- coding: utf-8 -*-
"""
Color Transfer Neural Network (Colornet).
Created on Wed Oct 10 17:00:00 2018
Author: Prasun Roy | https://prasunroy.github.io
GitHub: https://github.com/prasunroy/stefann

"""


# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import glob
import numpy
import os
import tensorflow

from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers import LeakyReLU
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from PIL import Image

from utils import TelegramIM


# configurations
# -----------------------------------------------------------------------------
TIMESTAMP        = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

RANDOM_SEED      = None

ARCHITECTURE     = 'colornet'

TRAIN_IMAGES_DIR = 'datasets/colornet/train/'
VALID_IMAGES_DIR = 'datasets/colornet/valid/'
TEST_IMAGES_DIR  = 'datasets/colornet/test/'
OUTPUT_DIR       = 'output/{}/{}/'.format(ARCHITECTURE, TIMESTAMP)

IMAGE_FILE_EXT   = '.jpg'

FUNCTION_OPTIM   = 'adam'
FUNCTION_LOSS    = 'mae'

INPUT_IMAGE_SIZE = (64, 64)

SCALE_COEFF_IMG  = 1.
BATCH_SIZE       = 1
NUM_EPOCHS       = 1
NOTIFY_EVERY     = 1

AUTH_TOKEN       = None
CHAT_ID          = None
# -----------------------------------------------------------------------------


class Colornet(object):
    def __new__(self, input_shapes, optimizer, loss, weights=None):
        # build network
        x1 = Input(input_shapes[0])
        x2 = Input(input_shapes[1])
        
        y1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x1)
        y1 = LeakyReLU(alpha=0.2)(y1)
        y1 = BatchNormalization()(y1)
        
        y2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x2)
        y2 = LeakyReLU(alpha=0.2)(y2)
        y2 = BatchNormalization()(y2)
        
        y = Concatenate()([y1, y2])
        
        y = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = MaxPooling2D(pool_size=(2, 2))(y)
        
        y = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = MaxPooling2D(pool_size=(2, 2))(y)
        
        y = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        y = Conv2D(filters=256, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = UpSampling2D(size=(2, 2))(y)
        y = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        y = Conv2D(filters=3, kernel_size=(3, 3), padding='same')(y)
        y = LeakyReLU(alpha=0.2)(y)
        
        # compile network
        model = Model(inputs=[x1, x2], outputs=y)
        model.compile(optimizer=optimizer, loss=loss)
        
        # optionally load existing weights into network
        try:
            if not weights is None:
                model.load_weights(weights)
        except:
            pass
        
        return model
