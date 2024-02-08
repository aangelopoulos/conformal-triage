import os
import datetime
import copy
import pandas as pd
import itertools
from serialize_model import serialize_model
from evaluate import evaluate, load_datasets
import pdb

if __name__ == "__main__":
    model_strs = ['isotonic', 'logistic', 'threshold']
    desired_ppvs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    desired_npvs = [0.9, 0.925, 0.95, 0.96, 0.97]
    tolerance = 0.2
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    n_train = 1000
    n_val = 1000

    # For each combination of desired ppv and npv, using itertools
    df_list = []
    # Evaluate hand-designed rule
    sgmds, labels, sgmds_small, labels_small = load_datasets()
    metrics = evaluate(sgmds, labels, None)
    metrics_small = evaluate(sgmds_small, labels_small, None)
    metrics_small = {k+'_small': v for k, v in metrics_small.items()} # postfix with _small
    metrics.update(metrics_small)
    metrics['desired_ppv'] = None
    metrics['desired_npv'] = None
    metrics['model'] = 'hand-designed'
    df_list.append(pd.DataFrame(metrics, index=[0]))

    for model_str, desired_ppv, desired_npv in itertools.product(model_strs, desired_ppvs, desired_npvs):
        metrics = serialize_model(model_str, desired_ppv, desired_npv, tolerance, date, n_train, n_val)
        metrics['model'] = model_str
        # pretty print the following : desired_ppv, desired_npv, metrics['ppv'], metrics['npv'], metrics['cls'].lambda_hi, metrics['cls'].lambda_lo
        print('model_str: {}, coverage: {:.2f}, desired_ppv: {:.2f}, desired_npv: {:.2f}, ppv: {:.2f}, npv: {:.2f}, lambda_hi: {:.2f}, lambda_lo: {:.2f}'.format(model_str, metrics['coverage'], desired_ppv, desired_npv, metrics['ppv'], metrics['npv'], metrics['cls'].lambda_hi, metrics['cls'].lambda_lo))
        df_list.append(pd.DataFrame(copy.deepcopy(metrics), index=[0]).drop(['cls', 'tolerance', 'date', 'n_train', 'n_val'], axis=1))


    df = pd.concat(df_list, axis=0, ignore_index=True)
    os.makedirs('../results', exist_ok=True)
    df.to_csv('../results/metrics.csv', index=False)
