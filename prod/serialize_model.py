import os
import numpy as np
import pickle as pkl
import pandas as pd
import datetime
import argparse
from .arch import ThresholdClassifierAbstention, IsotonicRegressionClassifierAbstention, LogisticRegressionClassifierAbstention
import pdb
from .evaluate import evaluate, string_to_array, load_datasets

def serialize_model(model_str, desired_ppv, desired_npv, tolerance, date, n_train, n_val, save=True):
    # Set randomness
    np.random.seed(0)

    # Load Data
    sgmds, labels, sgmds_small, labels_small = load_datasets()

    # Initialize, Train, and Calibrate Model
    if model_str == 'threshold':
        cls = ThresholdClassifierAbstention(verbose=False)
    elif model_str == 'isotonic':
        cls = IsotonicRegressionClassifierAbstention(verbose=False)
    elif model_str == 'logistic':
        cls = LogisticRegressionClassifierAbstention(verbose=False)
    else:
        raise ValueError(f'Unknown model_str: {model_str}')
    cls.fit(sgmds[:n_train], labels[:n_train])
    cls.calibrate(sgmds[:n_train+n_val], labels[:n_train+n_val], desired_ppv=desired_ppv, desired_npv=desired_npv, tolerance=tolerance)

    # Evaluate Model
    metrics = evaluate(sgmds[n_train:n_train+n_val], labels[n_train:n_train+n_val], cls=cls)
    metrics_small = evaluate(sgmds_small, labels_small, cls)

    # postfix '_small' to metrics_small keys
    metrics_small = {k+'_small': v for k, v in metrics_small.items()}
    metrics.update(metrics_small)

    # Save Model
    model_dict = {
        'cls': cls,
        'date': date,
        'desired_ppv': desired_ppv,
        'desired_npv': desired_npv,
        'tolerance': tolerance,
        'n_train': n_train,
        'n_val': n_val
    }
    model_dict.update(metrics)
    if save:
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./models/' + model_str, exist_ok=True)
        pkl.dump(model_dict, open(f'./models/{model_str}/{date}-{desired_ppv*100:.0f}-{desired_npv*100:.0f}-{tolerance*100:.0f}.pkl', 'wb'))
    return model_dict

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_str', type=str, default='isotonic')
    parser.add_argument('--desired_ppv', type=float, default=0.95)
    parser.add_argument('--desired_npv', type=float, default=0.7)
    parser.add_argument('--tolerance', type=float, default=0.2)
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_val', type=int, default=1000)
    args = parser.parse_args()

    serialize_model(args.model_str, args.desired_ppv, args.desired_npv, args.tolerance, datetime.datetime.now().strftime("%Y-%m-%d"), args.n_train, args.n_val)
