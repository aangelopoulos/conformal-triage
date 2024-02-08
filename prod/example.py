import os
import datetime
import copy
import pandas as pd
from evaluate import evaluate, load_datasets
from selective import selective_prediction
import pickle as pkl
import pdb

if __name__ == "__main__":
    # Load data
    sgmds, labels, sgmds_small, labels_small = load_datasets()

    # Example for AUTOMATED use case
    preds = selective_prediction(sgmds_small[:10])
    print(preds)

    # Example for MANUAL use case
    # Load model
    model = pkl.load(open('./models/curr_model.pkl', 'rb'))['cls']
    # Run evaluation
    metrics = evaluate(sgmds, labels, model)
    metrics_small = evaluate(sgmds_small, labels_small, model)
    metrics_small = {k+'_small': v for k, v in metrics_small.items()} # postfix with _small
    metrics.update(metrics_small)

    print(metrics)
