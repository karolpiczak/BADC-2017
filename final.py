#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Results averaging.

"""

import functools

import numpy as np
import pandas as pd

if __name__ == '__main__':

    ensemble_results = [
        pd.read_csv('run-A.csv'),
        pd.read_csv('run-A_ds.csv'),
        pd.read_csv('run-B.csv'),
        pd.read_csv('run-B_ds.csv'),
    ]

    ensemble_average = functools.reduce(lambda left, right: pd.merge(left, right, on='itemid'),
                                        ensemble_results)

    ensemble_average['mean'] = np.mean(ensemble_average.iloc[:, 1:], axis=1)
    ensemble_average = ensemble_average[['itemid', 'mean']]

    final_results = [
        ensemble_average,
        pd.read_csv('run-23.csv'),
        pd.read_csv('run-26.csv')
    ]

    final_average = functools.reduce(lambda left, right: pd.merge(left, right, on='itemid'),
                                     final_results)

    final_average['mean'] = np.mean(final_average.iloc[:, 1:], axis=1)
    final_average = final_average[['itemid', 'mean']]
    final_average.columns = [['itemid', 'hasbird']]

    final_average.to_csv('final.csv', index=False, float_format='%.3f')
