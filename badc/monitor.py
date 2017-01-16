# -*- coding: utf-8 -*-
"""Training progress reporting.

"""

import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import sklearn.metrics

from badc.generics import to_percentage


sb.set(style="white", palette="muted")


class Monitor(keras.callbacks.Callback):
    def __init__(self, model, validation_batch, run):
        self.model = model
        self.validation_batch = validation_batch
        self.run = str(run)
        self.loss = []
        self.train_score = []
        self.validation_score = []
        self.start_time = None

    def on_train_begin(self, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.train_score.append(logs.get('acc'))

        predictions = self.model.predict(self.validation_batch[0]).T[0]
        validation_score = sklearn.metrics.roc_auc_score(self.validation_batch[1], predictions)
        self.validation_score.append(validation_score)

        time_elapsed = time.time() - self.start_time
        self.start_time = time.time()

        self.plot_results(epoch, time_elapsed)

        print('Epoch {}: Loss: {} | Train: {} | Validation: {}'.format(
            epoch,
            np.round(self.loss[-1], 3),
            np.round(self.train_score[-1], 3),
            np.round(validation_score, 3)
        ))

        self.model.save_weights('results/run-' + str(self.run) + '.h5')

    def plot_results(self, epoch, time_elapsed):
        cmap = sb.diverging_palette(220, 10, as_cmap=True)
        cmap_delta = sb.diverging_palette(120, 10, as_cmap=True)

        x = range(len(self.loss))

        # History chart

        f = plt.figure(figsize=(12, 8))
        ax = f.gca()
        ax.plot(x, self.train_score, 'g-', self.validation_score, 'g--')
        ax.yaxis.set_ticks(np.arange(0.0, 1.01, 0.1))
        ax.set_xlim((0.0, epoch))
        ax.get_xaxis().set_visible(False)

        ax2 = ax.twinx()
        ax2.plot(x, self.loss, 'b--')

        plt.legend(['Train', 'Validation'], loc='upper left')
        title = 'Train: {}% / Val: {}% | Epoch {} | Loss {} | T/epoch {} s'
        plt.title(title.format(
            to_percentage(self.train_score[-1]),
            to_percentage(self.validation_score[-1]),
            epoch,
            np.round(self.loss[-1], 2),
            int(time_elapsed),
        ))

        plt.savefig('results/' + self.run + '-history.png', bbox_inches='tight')

        # Filters

        n_filters = np.shape(self.model.layers[0].get_weights()[0])[0]
        n_cols = 10
        n_rows = int(np.ceil(n_filters / 10))

        f = plt.figure(figsize=(10 // 2, n_rows // 2))

        for i in range(n_filters):
            row = int(i / n_cols)
            col = i % n_cols

            filter = self.model.layers[0].get_weights()[0][i, 0, :, :]
            delta = self.model.layers[0].get_weights()[0][i, 1, :, :]

            ax = plt.subplot2grid((n_rows, n_cols * 2), (row, col * 2))
            ax.imshow(filter, origin='lower', aspect='auto', cmap=cmap, interpolation='nearest')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot2grid((n_rows, n_cols * 2), (row, col * 2 + 1))
            ax.imshow(delta, origin='lower', aspect='auto', cmap=cmap_delta, interpolation='nearest')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig('results/' + self.run + '-filters.png', bbox_inches='tight')

        plt.close('all')
