import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
import pdb

class ThresholdClassifierAbstention():
    def __init__(self, verbose=False, num_thresh=5000):
        self.verbose = verbose
        self.k_hi = -1
        self.k_lo = -1
        self.lambda_hi = -1
        self.lambda_lo = -1
        self.desired_ppv = -1
        self.desired_npv = -1
        self.tolerance = 0.2
        self.num_thresh = num_thresh
        self.min_data = [500, 50]

    def fit(self, sgmds, labels):
        sgmds2 = sgmds.copy()
        sgmds2.sort()
        sgmds2 = sgmds2[:,::-1]
        best_p_metric = 0
        best_n_metric = 0
        # Try all combinations of k_hi and k_lo (they are independent)
        # If an example has at least k_hi scores above a (large) threshold, it is positive.
        # If it has at most k_lo scores above a (small) threshold, it is negative.
        # The strategy is to fix the fraction of abstentions (e.g., at 10% or 50%) and then choose k_hi and k_lo to maximize PPV+TPR and NPV+TNR, respectively.
        for k in range(1, min(sgmds2.shape[1],5)):
            lower_quantile = 0.6
            upper_quantile = 0.9
            if lower_quantile == upper_quantile:
                continue
            positives = sgmds2[:,k] >= upper_quantile
            negatives = (np.sum(sgmds2 >= lower_quantile, axis=1) <= k) & (~positives)
            curr_ppv = (positives & labels).sum()/positives.sum() if positives.sum() > 0.05*labels.sum() else 0
            curr_tpr = (positives & labels).sum()/labels.sum()
            curr_p_metric = curr_ppv + curr_tpr
            curr_npv = (negatives & (~labels)).sum()/negatives.sum() if negatives.sum() > 0.05*(1-labels).sum() else 0
            curr_tnr = (negatives & (~labels)).sum()/(1-labels).sum()
            curr_n_metric = curr_npv + curr_tnr
            if self.verbose:
                print(f"k={k} | p_metric/ppv/tpr={curr_p_metric:.3f}/{curr_ppv:.3f}/{curr_tpr:.3f} | n_metric/npv/tnr={curr_n_metric}/{curr_npv:.3f}/{curr_tnr:.3f} | best_k_thresh=({self.k_lo},{self.k_hi})")
            if curr_p_metric > best_p_metric:
                self.k_hi = k
                best_p_metric = curr_p_metric
            if curr_n_metric > best_n_metric:
                self.k_lo = k
                best_n_metric = curr_n_metric

    def calibrate(self, cal_sgmds, cal_labels, desired_ppv=0.6, desired_npv=0.95, tolerance=0.2):
        if self.k_hi == -1 or self.k_lo == -1:
            raise Exception("Must call fit() before calibrate()")
        # Get model probabilities
        cal_phats_lo = self.predict_proba(cal_sgmds)[:,0]
        cal_phats_hi = self.predict_proba(cal_sgmds)[:,1]
        # Set up grid of lambdas for LTT
        lambdas = np.linspace(0,1,self.num_thresh)

        # Calibrate lambda_hi to achieve ppv guarantee
        #if desired_ppv == 0.7:
        #    pdb.set_trace()

        def selective_frac_nonich(lam): return 1-cal_labels[cal_phats_hi > lam].sum()/(cal_phats_hi > lam).sum()
        def nlambda(lam): return (cal_phats_hi > lam).sum()
        lambdas_highrisk = np.array([lam for lam in lambdas if nlambda(lam) >= self.min_data[1]]) # Make sure there's some data in the top bin.
        def invert_for_ub(r,lam): return binom.cdf(selective_frac_nonich(lam)*nlambda(lam),nlambda(lam),r)-tolerance
# Construct upper bound
        def selective_risk_ub(lam): return brentq(invert_for_ub,0,0.9999,args=(lam,))
# Scan to choose lamabda hat
        for lhat_highrisk in np.flip(lambdas_highrisk):
            if selective_risk_ub(lhat_highrisk-1/lambdas_highrisk.shape[0]) > (1-desired_ppv): break
        if lhat_highrisk == lambdas_highrisk[-1]:
            lhat_highrisk = 1 # Could not find a lambda that achieves the desired ppv, so just set it to 1.

        # Calibrate lambda_lo to achieve npv guarantee
        def selective_frac_ich(lam): return cal_labels[cal_phats_lo <= lam].sum()/(cal_phats_lo <= lam).sum()
        def nlambda(lam): return (cal_phats_lo <= lam).sum()
        lambdas_lowrisk = np.array([lam for lam in lambdas if nlambda(lam) >= self.min_data[0]]) # Make sure there's some data in the top bin.
        def invert_for_ub(r,lam): return binom.cdf(selective_frac_ich(lam)*nlambda(lam),nlambda(lam),r)-tolerance
        # Construct upper bound
        def selective_risk_ub(lam): return brentq(invert_for_ub,0,0.9999,args=(lam,))
        # Scan to choose lambda hat
        for lhat_lowrisk in lambdas_lowrisk:
            if selective_risk_ub(lhat_lowrisk+1/lambdas_lowrisk.shape[0]) > (1-desired_npv) : break
        if lhat_lowrisk == lambdas_lowrisk[0]:
            lhat_lowrisk = 0 # Could not find a lambda that achieves the desired npv, so just set it to 1.

        self.lambda_hi = lhat_highrisk
        self.lambda_lo = lhat_lowrisk
        self.desired_ppv = desired_ppv
        self.desired_npv = desired_npv
        self.tolerance = tolerance

    def predict(self, sgmds):
        """
        Returns a vector of predictions, where 'P' means positive, 'N' means negative, and 'U' means uncertain.
        """
        if self.k_hi == -1 or self.k_lo == -1:
            raise Exception("Must call fit() before predict()")
        if self.lambda_hi == -1 or self.lambda_lo == -1:
            raise Exception("Must call calibrate() before predict()")
        sgmds2 = sgmds.copy()
        sgmds2.sort()
        sgmds2 = sgmds2[:,::-1]
        return np.array(['P' if sgmds2[i,self.k_hi] >= self.lambda_hi else 'N' if (sgmds2[i,:self.k_lo] <= self.lambda_lo).all() else 'U' for i in range(sgmds2.shape[0])])

    def predict_proba(self, sgmds):
        if self.k_hi == -1 or self.k_lo == -1:
            raise Exception("Must call fit() before predict_proba()")
        sgmds2 = sgmds.copy()
        sgmds2.sort()
        sgmds2 = sgmds2[:,::-1]
        return np.stack([sgmds2[:,self.k_lo], sgmds2[:,self.k_hi]], axis=1)
