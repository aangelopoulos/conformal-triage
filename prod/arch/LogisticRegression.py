import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from .ltt_calibration import calibrate
import pdb


class LogisticRegressionClassifierAbstention():
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model = LogisticRegression()
        self.lambda_hi = None
        self.lambda_lo = None
        self.desired_ppv = None
        self.desired_npv = None
        self.tolerance = None

    @staticmethod
    def featurize(sgmds):
        sorted_sgmds = np.sort(sgmds, axis=1)[:,::-1]
        return sorted_sgmds[:,:4]

    def fit(self, sgmds, labels):
        self.model = self.model.fit(self.featurize(sgmds), labels)

    def calibrate(self, cal_sgmds, cal_labels, desired_ppv=0.6, desired_npv=0.95, tolerance=0.2):
        # Get model probabilities
        cal_phats_lo = self.predict_proba(cal_sgmds)[:,0]
        cal_phats_hi = self.predict_proba(cal_sgmds)[:,1]

        # Calibrate lambdas to achieve ppv and npv guarantees
        min_data = [500,50]
        num_thresh = 5000
        lhat_lowrisk, lhat_highrisk = calibrate(cal_phats_lo, cal_phats_hi, cal_labels, desired_ppv, desired_npv, tolerance, num_thresh, min_data)
        self.lambda_hi = lhat_highrisk
        self.lambda_lo = lhat_lowrisk
        self.desired_ppv = desired_ppv
        self.desired_npv = desired_npv
        self.tolerance = tolerance

    def predict(self, sgmds):
        """
        Returns a vector of predictions, where 'P' means positive, 'N' means negative, and 'U' means uncertain.
        """
        phats = self.predict_proba(sgmds)
        preds = np.array(['U']*phats.shape[0])
        preds[phats[:,0] >= self.lambda_lo] = 'N'
        preds[phats[:,1] >= self.lambda_hi] = 'P'
        return preds

    def predict_proba(self, sgmds):
        """
        Returns a matrix of probabilities, where the first column is the probability of a negative prediction, the second column is the probability of a positive prediction, and the third column is the probability of abstention.
        """
        phats = self.model.predict_proba(self.featurize(sgmds))
        return phats
