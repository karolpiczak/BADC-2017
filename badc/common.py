# -*- coding: utf-8 -*-
"""Common imports and data loading.

"""

import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import librosa
import numpy as np
import pandas as pd
import pydub
import sklearn as sk

from tqdm import *

DATA_PATH = '/volatile/BADC/'

os.environ['KERAS_BACKEND'] = 'theano'

import keras
keras.backend.set_image_dim_ordering('th')
from keras.layers.convolutional import Convolution2D as Conv
from keras.layers.convolutional import MaxPooling2D as Pool
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.callbacks import LearningRateScheduler, Callback
from keras.layers.normalization import BatchNormalization


L1 = keras.regularizers.l1
L2 = keras.regularizers.l2

freefield = pd.read_csv(DATA_PATH + 'ff1010bird_metadata.csv')
warblr = pd.read_csv(DATA_PATH + 'warblrb10k_public_metadata.csv')

freefield['dataset'] = 'freefield'
warblr['dataset'] = 'warblr'

try:
    os.mkdir(DATA_PATH + 'resampled')
    for recording in sorted(os.listdir(DATA_PATH + 'train')):
        if os.path.isfile(DATA_PATH + 'train/' + recording):
            subprocess.run(['sox',
                            '-S', DATA_PATH + 'train/' + recording,
                            '-r', '44100',
                            '-b', '16',
                            DATA_PATH + 'resampled/' + recording])
except OSError:
    pass

try:
    os.mkdir(DATA_PATH + 'resampled-test')
    for recording in sorted(os.listdir(DATA_PATH + 'test')):
        if os.path.isfile(DATA_PATH + 'test/' + recording):
            subprocess.run(['sox',
                            '-S', DATA_PATH + 'test/' + recording,
                            '-r', '44100',
                            '-b', '16',
                            DATA_PATH + 'resampled-test/' + recording])
except OSError:
    pass
