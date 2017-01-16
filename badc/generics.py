# -*- coding: utf-8 -*-
"""Generic helper functions.

"""

from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import pydub


def load_audio(path):
    audio = pydub.AudioSegment.from_file(path).set_frame_rate(44100).set_channels(1)
    raw = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)   # convert to float
    return raw


def to_percentage(number):
    return int(number * 1000) / 10.0


def describe(model):
    description = 'Model layers / shapes / parameters:\n'
    total_params = 0

    for layer in model.layers:
        layer_params = layer.count_params()
        description += '- {}'.format(layer.name).ljust(20)
        description += '{}'.format(layer.input_shape).ljust(20)
        description += '{0:,}'.format(layer_params).rjust(12)
        description += '\n'
        total_params += layer_params

    description += 'Total:'.ljust(30)
    description += '{0:,}'.format(total_params).rjust(22)

    print(description)


class LearningRateDecay(Callback):
    def __init__(self, epochs, decay):
        super().__init__()
        self.epochs = epochs
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.epochs == 0:
            K.set_value(self.model.optimizer.lr, self.decay * self.model.optimizer.lr.get_value())
