import numpy as np
import pickle as pkl

def selective_prediction( sigmoid_scores, method='auto' ):
    """
    Selective prediction for sigmoid scores.

    Parameters
    ----------
    sigmoid_scores : array-like, shape (n_samples, n_slices)
        The sigmoid scores representing the probability of ICH for each CT slice of each sample.
        These scores lie in [0,1], and do not normalize to 1.
    method : string, optional (default='auto')
        The method to use for selective prediction. Options are 'auto' and 'hand-designed-rule'.

    Returns
    -------
    predictions : array-like, shape (n_samples,)
        The selective predictions for each sample. Options are 'P' (ICH-Positive), 'N' (ICH-Negative), and 'U' (ICH-Uncertain).
    """
    if method == 'auto':
        return selective_prediction_auto( sigmoid_scores )
    elif method == 'hand-designed-rule':
        return selective_prediction_hand_designed_rule( sigmoid_scores )
    else:
        raise ValueError( 'Invalid method: {}'.format( method ) )

def selective_prediction_auto( sigmoid_scores, ):
    """
    Selective ICH prediction for sigmoid scores using the 'auto' method.
    This uses selective classification by Learn then Test (LTT) to predict ICH-Positive ('P'), ICH-Negative ('N'), or ICH-Uncertain ('U') such that the positive predictive value and negative predictive value are both guaranteed to be above the user-specified arguments.

    Parameters
    ----------
    sigmoid_scores : array-like, shape (n_samples, n_slices)
        The sigmoid scores representing the probability of ICH for each CT slice of each sample.
        These scores lie in [0,1], and do not normalize to 1.

    Returns
    -------
    predictions : array-like, shape (n_samples,)
        The selective predictions for each sample. Options are 'P' (ICH-Positive), 'N' (ICH-Negative), and 'U' (ICH-Uncertain).
    """
    with open('models/curr_model.pkl', 'rb') as f: # model string is models/{model_str}-{date}-{desired_ppv}-{desired_npv}-{tolerance}.pkl
        model = pkl.load(f)
    return model['cls'].predict(sigmoid_scores)

def selective_prediction_hand_designed_rule( sigmoid_scores ):
    """
    Selective prediction for sigmoid scores using hand-designed rule.
    If any two slices have scores greater than 0.9, then the sample is predicted as ICH-Positive. ('P')
    If no more than three slices have scores above 0.6, then the sample is predicted as ICH-Negative. ('N')
    Otherwise, the sample is predicted as ICH-Uncertain. ('U')

    Parameters
    ----------
    sigmoid_scores : array-like, shape (n_samples, n_slices)
        The sigmoid scores representing the probability of ICH for each CT slice of each sample.
        These scores lie in [0,1], and do not normalize to 1.

    Returns
    -------
    predictions : array-like, shape (n_samples,)
        The selective predictions for each sample. Options are 'P' (ICH-Positive), 'N' (ICH-Negative), and 'U' (ICH-Uncertain).
    """
    predictions_highrisk = ((sigmoid_scores >= 0.9).sum(axis=1) >= 3).astype(int)
    predictions_lowrisk = ((sigmoid_scores >= 0.6).sum(axis=1) <= 1).astype(int)
    predictions = np.where( predictions_highrisk, 'P', np.where( predictions_lowrisk, 'N', 'U' ) )
    return predictions
