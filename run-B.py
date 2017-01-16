#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""High resolution spectrogram convnet.

Random seed #2.

"""

import os

os.environ['THEANO_FLAGS'] = ('floatX=float32,'
                              'device=gpu1,'
                              'dnn.conv.algo_bwd_filter=deterministic,'
                              'dnn.conv.algo_bwd_data=deterministic')

from badc.common import *
from badc.generics import describe, LearningRateDecay
from badc.dataset import iterbatches, generate_predictions

import badc.monitor


if __name__ == '__main__':
    RUN = 'B'
    np.random.seed(220170113)

    train = pd.concat([freefield, warblr], ignore_index=True)
    train = train.iloc[np.random.permutation(len(train))]

    validation = train[:1000]
    # train = train[1000:]

    model = keras.models.Sequential()

    model.add(Conv(80, 165, 8, init='he_uniform', W_regularizer=L2(0.001), input_shape=(2, 170, 950)))
    model.add(LeakyReLU())
    model.add(Pool((6, 4)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(160, 1, 2, W_regularizer=L2(0.001), init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(240, 1, 2, W_regularizer=L2(0.001), init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(320, 1, 2, W_regularizer=L2(0.001), init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(400, 1, 2, W_regularizer=L2(0.001), init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Dropout(0.25))

    model.add(Conv(1, 1, 1, init='he_uniform'))
    model.add(Activation('sigmoid'))

    model.add(GlobalMaxPooling2D())

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    describe(model)

    if os.path.exists('results/run-' + str(RUN) + '.h5'):
        model.load_weights('results/run-' + str(RUN) + '.h5')
    else:
        validation_batch = next(iterbatches(1000, validation, augment=False, hires=True))
        monitor = badc.monitor.Monitor(model, validation_batch, RUN)

        learning_rate_decay = LearningRateDecay(100, 0.5)

        model.fit_generator(generator=iterbatches(32, train, augment=True, hires=True),
                            samples_per_epoch=len(train),
                            nb_epoch=50,
                            callbacks=[monitor, learning_rate_decay],
                            verbose=0,
                            max_q_size=10)

        model.save_weights('results/run-' + str(RUN) + '.h5')

    generate_predictions(model, RUN, hires=True)
