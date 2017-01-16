# -*- coding: utf-8 -*-
"""Main data processing for the BADC dataset.

"""

import os

import librosa
import numpy as np
import pandas as pd
import scipy.signal
import skimage as skim
import skimage.measure
from tqdm import *

from badc.common import DATA_PATH
from badc.generics import load_audio


def generate_delta(spec):
    # ported librosa v0.3.1. implementation
    window = np.arange(4, -5, -1)
    padding = [(0, 0), (5, 5)]
    delta = np.pad(spec, padding, mode='edge')
    delta = scipy.signal.lfilter(window, 1, delta, axis=-1)
    idx = [Ellipsis, slice(5, -5, None)]
    return delta[idx]


def generate_spec(recording, hires, test=False):
    hires_ext = '.hires' if hires else ''
    if test:
        audio = load_audio(DATA_PATH + 'resampled-test/' + recording + '.wav')
    else:
        audio = load_audio(DATA_PATH + 'resampled/' + recording + '.wav')

    spec = librosa.feature.melspectrogram(audio, sr=44100, n_fft=2205, hop_length=441, n_mels=180,
                                          fmax=16000)
    freqs = librosa.core.mel_frequencies(n_mels=180, fmax=16000)
    spec = librosa.core.perceptual_weighting(spec, freqs, ref_power=np.max)
    if not hires:
        spec = skim.measure.block_reduce(spec, block_size=(3, 2), func=np.mean)
        spec = spec[5:55, :500]
    else:
        spec = spec[5:175, :1000]
    spec += 40
    spec /= 10.0
    if test:
        np.save(DATA_PATH + 'resampled-test/' + recording + hires_ext + '.spec.npy',
                spec.astype('float16'), allow_pickle=False)
    else:
        np.save(DATA_PATH + 'resampled/' + recording + hires_ext + '.spec.npy',
                spec.astype('float16'), allow_pickle=False)


def load_spec(recording, hires, test=False):
    hires_ext = '.hires' if hires else ''
    recording = str(recording)
    if test:
        spec_file = DATA_PATH + 'resampled-test/' + recording + hires_ext + '.spec.npy'
    else:
        spec_file = DATA_PATH + 'resampled/' + recording + hires_ext + '.spec.npy'
    if not os.path.exists(spec_file):
        generate_spec(recording, hires, test)
    return np.load(spec_file).astype('float32')


def augment_spec(spec):
    augmentation_level = abs(np.random.normal())
    spec += np.random.normal(0, augmentation_level, np.shape(spec))
    return spec


def shuffle_spec(spec):
    segments = 10
    spec = np.array(np.split(spec, segments, 1))
    spec = spec[np.random.permutation(len(spec)), :]
    return np.concatenate(spec, axis=1)


def _iterrows(dataset):
    while True:
        for row in dataset.iloc[np.random.permutation(len(dataset))].itertuples():
            yield row


def iterbatches(batch_size, dataset, augment=False, hires=False):
    itrain = _iterrows(dataset)
    count = 0

    while True:
        X, y = [], []

        for i in range(batch_size):
            row = next(itrain)
            spec = load_spec(row.itemid, hires)
            if not hires:
                spec.resize((50, 500))
                offset = np.random.randint(21)
                spec = spec[:, (0 + offset):(480 + offset)]
            else:
                spec.resize((170, 1000))
                offset = np.random.randint(51)
                spec = spec[:, (0 + offset):(950 + offset)]

            hasbird = row.hasbird
            if augment:
                spec = shuffle_spec(spec)
                spec = augment_spec(spec)
                if np.random.random() < 0.25:
                    row2 = next(itrain)
                    spec2 = load_spec(row2.itemid, hires)

                    if not hires:
                        spec2.resize((50, 500))
                        offset2 = np.random.randint(21)
                        spec2 = spec2[:, (0 + offset2):(480 + offset2)]
                    else:
                        spec2.resize((170, 1000))
                        offset2 = np.random.randint(51)
                        spec2 = spec2[:, (0 + offset2):(950 + offset2)]

                    spec2 = shuffle_spec(spec2)
                    spec2 = augment_spec(spec2)
                    spec = np.mean([spec, spec2], axis=0)
                    hasbird = int(row.hasbird or row2.hasbird)
            X.append(np.stack([spec, generate_delta(spec)]))
            y.append(hasbird)
            count += 1

        X = np.stack(X)
        y = np.array(y)

        yield X, y


def generate_predictions(model, run, hires):
    test_files = os.listdir(DATA_PATH + '/resampled-test')

    test_names = []
    test_X = []
    test_predictions = []

    for file in tqdm(test_files):
        if file[-4:] != '.wav':
            continue
        name = file[:-4]
        spec = load_spec(name, hires=hires, test=True)

        if hires:
            spec.resize((170, 1000))
            spec = spec[:, 25:975]
        else:
            spec.resize((50, 500))
            spec = spec[:, 10:490]

        test_X.append(np.stack([spec, generate_delta(spec)]))
        test_names.append(name)

        if len(test_X) > 1000:
            test_X = np.stack(test_X)
            predictions = model.predict(test_X)
            predictions = [prediction[0] for prediction in predictions]
            predictions = np.round(predictions, 3)
            test_predictions.extend(predictions)
            test_X = []

    test_X = np.stack(test_X)
    predictions = model.predict(test_X)
    predictions = [prediction[0] for prediction in predictions]
    predictions = np.round(predictions, 3)
    test_predictions.extend(predictions)

    results = pd.DataFrame({'itemid': test_names, 'hasbird': test_predictions},
                           columns=['itemid', 'hasbird'])
    results = results.sort_values('itemid')
    results.to_csv('run-' + str(run) + '.csv', index=False, float_format='%.3f')
