import numpy as np
import pandas as pd
import pickle as pkl
import datetime
import argparse
from .arch import ThresholdClassifierAbstention
from .selective import selective_prediction
import pdb

def string_to_array(lst, arr_len):
    import numpy as np

    # create an empty list to hold the arrays
    arrays = np.zeros((len(lst), arr_len))

    # iterate over the list
    for j in range(len(lst)):
        string = lst[j]

        # split the string at the colon
        slices = string.split('\n')
        num_slices = len(slices)

        for i in range(num_slices):
            if slices[i] == '':
                continue
            # get the index from the first part, subtract 1 to make it 0-indexed
            parts = slices[i].split(':')
            index = int(parts[0]) - 1

            # get the number from the second part, removing the parentheses and converting to float
            # strip any newline characters before converting
            number = float(parts[1].strip()[1:-1].replace('\r', '').replace('\n', '').replace(')','').replace('(',''))

            arrays[j,index] = number

    return arrays

def evaluate(sgmds, labels, cls=None):
    if cls is None:
        preds = selective_prediction(sgmds, method='hand-designed-rule')
    else:
        preds = cls.predict(sgmds)
    # Calculate metrics
    fn = ((preds == 'N') & labels).sum()
    fp = ((preds == 'P') & ~labels).sum()
    tp = ((preds == 'P') & labels).sum()
    tn = ((preds == 'N') & ~labels).sum()
    ppv = labels[preds == 'P'].mean() if fp+tp > 0 else 1.0
    npv = 1-labels[preds == 'N'].mean() if fn+tn > 0 else 1.0
    coverage = 1-(preds == 'U').mean() if (preds == 'U').sum() > 0 else 1.0
    # Create a dictionary of the results
    metrics = {'ppv': ppv, 'npv': npv, 'coverage': coverage, 'fn': fn, 'fp': fp, 'tp': tp, 'tn': tn}
    return metrics

def load_datasets(ids=False):
    # Load Data
    data = np.load('../data/proc-ich-me-data.npz')
    shuffler = np.random.permutation(data['labels'].shape[0])
    sgmds = data['slice_probs'][shuffler,:]
    labels = data['labels'].astype(bool)[shuffler]
    if ids:
        parsing_ids = data['parsing_id'][shuffler]
        diagnostic_ids = data['diagnostic_id'][shuffler]

    # Load Data (Small dataset)
    data_small = np.load('../data/proc-small-clean.npz')
    shuffler_small = np.random.permutation(data_small['labels'].shape[0])
    sgmds_small = data_small['slice_probs'][shuffler_small,:]
    labels_small = data_small['labels'].astype(bool)[shuffler_small]

    if ids:
        return sgmds, labels, parsing_ids, diagnostic_ids, sgmds_small, labels_small
    else:
        return sgmds, labels, sgmds_small, labels_small
