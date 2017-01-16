#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low resolution spectrogram convnet.

When compared to A_ds/B_ds:
- Less filters
- No L2 regularization
- More epochs
- Faster learning rate decay
- Less dropout

- Originally used shuffle_spec() with 5 segments (spec length of 475) and less noise augmentations

"""

import os

os.environ['THEANO_FLAGS'] = ('floatX=float32,'
                              'device=gpu0,'
                              'dnn.conv.algo_bwd_filter=deterministic,'
                              'dnn.conv.algo_bwd_data=deterministic')

from badc.common import *
from badc.generics import describe, LearningRateDecay
from badc.dataset import iterbatches, generate_predictions

import badc.monitor


if __name__ == '__main__':
    RUN = '23'
    np.random.seed(20161212)

    train = pd.concat([freefield, warblr], ignore_index=True)
    train = train.iloc[np.random.permutation(len(train))]

    validation = train[:1000]
    train = train[1000:]

    model = keras.models.Sequential()

    model.add(Conv(40, 47, 6, init='he_uniform', input_shape=(2, 50, 480)))
    model.add(LeakyReLU())
    model.add(Pool((4, 3)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(80, 1, 2, init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(120, 1, 2, init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(160, 1, 2, init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Conv(200, 1, 2, init='he_uniform'))
    model.add(LeakyReLU())
    model.add(Pool((1, 2)))
    model.add(BatchNormalization(axis=1))

    model.add(Dropout(0.2))

    model.add(Conv(1, 1, 1, init='he_uniform'))
    model.add(Activation('sigmoid'))

    model.add(GlobalMaxPooling2D())

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.005, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    describe(model)

    if os.path.exists('results/run-' + str(RUN) + '.h5'):
        model.load_weights('results/run-' + str(RUN) + '.h5')
    else:
        validation_batch = next(iterbatches(1000, validation, augment=False, hires=False))
        monitor = badc.monitor.Monitor(model, validation_batch, RUN)

        learning_rate_decay = LearningRateDecay(50, 0.5)

        model.fit_generator(generator=iterbatches(32, train, augment=True, hires=False),
                            samples_per_epoch=len(train),
                            nb_epoch=500,
                            callbacks=[monitor, learning_rate_decay],
                            verbose=0,
                            max_q_size=10)

        model.save_weights('results/run-' + str(RUN) + '.h5')

    generate_predictions(model, RUN, hires=False)
