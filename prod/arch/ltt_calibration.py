import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
import pdb

def calibrate(cal_phats_lo, cal_phats_hi, cal_labels, desired_ppv, desired_npv, tolerance, num_thresh, min_data):
    # Set up grid
    lambdas = np.linspace(0.01,0.99,num_thresh)
    # Calibrate lambda_hi to achieve ppv guarantee
    def selective_frac_nonich(lam): return 1-cal_labels[cal_phats_hi > lam].sum()/(cal_phats_hi > lam).sum()
    def nlambda(lam): return (cal_phats_hi > lam).sum()
    lambdas_highrisk = np.array([lam for lam in lambdas if nlambda(lam) >= min_data]) # Make sure there's some data in the top bin.
    def invert_for_ub(r,lam): return binom.cdf(selective_frac_nonich(lam)*nlambda(lam),nlambda(lam),r)-tolerance
# Construct upper bound
    def selective_risk_ub(lam): return brentq(invert_for_ub,0,0.9999,args=(lam,))
# Scan to choose lamabda hat
    for lhat_highrisk in np.flip(lambdas_highrisk):
        sr = selective_risk_ub(lhat_highrisk-1/lambdas_highrisk.shape[0])
        if sr > (1-desired_ppv): break
    if lhat_highrisk == lambdas_highrisk[-1]:
        lhat_highrisk = 1 # Could not find a lambda that achieves the desired ppv, so just set it to 1.

    # Calibrate lambda_lo to achieve npv guarantee
    def selective_frac_ich(lam): return cal_labels[cal_phats_lo > lam].sum()/(cal_phats_lo > lam).sum()
    def nlambda(lam): return (cal_phats_lo > lam).sum()
    lambdas_lowrisk = np.array([lam for lam in lambdas if nlambda(lam) >= 2*min_data]) # Make sure there's some data in the top bin.
    def invert_for_ub(r,lam): return binom.cdf(selective_frac_ich(lam)*nlambda(lam),nlambda(lam),r)-tolerance
    # Construct upper bound
    def selective_risk_ub(lam): return brentq(invert_for_ub,0,0.9999,args=(lam,))
    # Scan to choose lambda hat
    for lhat_lowrisk in np.flip(lambdas_lowrisk):
        sr = selective_risk_ub(lhat_lowrisk+1/lambdas_lowrisk.shape[0])
        if sr > (1-desired_npv) : break
    if lhat_lowrisk == lambdas_lowrisk[-1]:
        lhat_lowrisk = 1 # Could not find a lambda that achieves the desired npv, so just set it to 1.
    return lhat_lowrisk, lhat_highrisk
