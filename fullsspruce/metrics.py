import pandas as pd
import numpy as np
import seaborn as sns

from scipy.stats import t, linregress

DEFAULT_DP4_PARAMETERS = {
    '1H': {
        'nu': 14.18,
        'std': 0.185
    },
    '13C': {
        'nu': 11.38, 
        'std': 2.306
    }
}

def compute_stats(g, mol_id_field='mol_id', return_per_mol=False,
                  tgt_field='pred_mu'):
    """
    Compute statistics for groups g

    foodf.grouby(bar).apply(compute_stats)

    """
    per_mol = g.groupby( mol_id_field)\
                .agg({'delta_abs' : 'mean', 
                    'delta' : lambda x: np.sqrt(np.mean(x**2)), })\
                    .rename(columns={'delta_abs' : 'mol_MAE', 
                                        'delta' : 'mol_MSE'})
    b = g.groupby(mol_id_field).apply(lambda x: np.sqrt(np.mean(np.sort(x['value']) - np.sort(x[tgt_field]))**2))
    per_mol['sorted_mol_MSE'] = b

    b = g.groupby(mol_id_field).apply(lambda x: np.mean(np.abs(np.sort(x['value']) - np.sort(x[tgt_field]))))
    per_mol['sorted_mol_MAE'] = b

    res = per_mol.mean()
    res['mean_abs'] = g.delta_abs.mean()
    res['std'] = g.delta.std()
    res['n'] = len(g)
    res['mol_n'] = len(g[mol_id_field].unique())
    
    if return_per_mol:
        return res, per_mol
    else:
        return res


def sorted_bin_stats(conf, data, BIN_N=None, aggfunc = np.mean, bins=None):
    """
    Useful for sorting the data by confidence interval and binning. 

    imagine you want to plot error in estimate as a function of 
    confidence interval. you could just plot the rolling mean, 
    but at the very low confidence intervals you'll have very few
    points and so you'll get a high-variance estimator. 

    this basically bins the confidence regions such that you always
    have the same # of datapoints in each bin

    
    """
    conf = np.array(conf)
    data = np.array(data)
    sort_idx = np.argsort(conf)
    conf = conf[sort_idx]
    data = data[sort_idx]

    if bins is None:
        bins = np.linspace(0.0, 1.0, BIN_N)
    else:
        BIN_N = len(bins)
    tholds = np.array([conf[min(len(conf)-1, int(i*len(conf)))] for i in bins[1:]])

    m = np.array([aggfunc(data[conf <= tholds[i]]) for i in range(BIN_N-1)])

    frac_data = np.linspace(0.0, 1.0, len(m))
    return m, tholds, frac_data

def sorted_bin_boxplot(conf, data, BIN_N, TGT_THOLDS):
    """
    Useful for sorting the data by confidence interval and binning. 

    imagine you want to plot min/max of error in estimate as a function of 
    confidence interval. you could just plot the rolling min/max, 
    but at the very low confidence intervals you'll have very few
    points and so you'll get a high-variance estimator. 

    this basically bins the confidence regions such that you always
    have the same # of datapoints in each bin

    
    """
    conf = np.array(conf)
    data = np.array(data)
    sort_idx = np.argsort(conf)
    conf = conf[sort_idx]
    data = data[sort_idx]

    bins = np.linspace(0.0, 1.0, BIN_N)
    tholds = [conf[min(len(conf)-1, int(i*len(conf)))] for i in bins[1:]]

    m = [data[conf <= tholds[i]] for i in range(BIN_N-1)]
    frac_data = np.linspace(0.0, 1.0, len(m))
    indices = np.searchsorted(frac_data, TGT_THOLDS)

    df = pd.DataFrame()
    for i,index in enumerate(indices):
        for z in m[index]:
            df = df.append({'error':z,'frac_data':TGT_THOLDS[i]},ignore_index=True)
    print(df)
    ax = sns.boxplot(x='frac_data',y='error',data=df)
    fig = ax.get_figure()
    return fig

def calculate_DP4(comp, pred, dp4_params=DEFAULT_DP4_PARAMETERS):
    """
    Take in the experimental or comparison result and get the DP4 unnormalized product of 
    probabilities using the given dp4_params for the T distribution. 

    Inputs:
        comp: Dictionary with entry for 1H and 13C. Each entry is a dictionary mapping from 
            atom index to value.
        pred: Dictionary with entry for 1H and 13C. Each entry is a dictionary mapping from 
            atom index to value.
        dp4_params: DP4 parameters to use to set up the T distribution (defaults provided).

    Returns:
        (1H_dp4, 13C_dp4): The DP4 probability of the protons and carbons, respectively, in 
            log space
    """
    # First, collect the shifts as lists
    d_exp_1H, d_calc_1H, d_exp_13C, d_calc_13C = [], [], [], []
    for atom_idx, value in comp['1H'].items():
        d_calc_1H += [pred['1H'][atom_idx]]
        d_exp_1H += [value]
    for atom_idx, value in comp['13C'].items():
        d_calc_13C += [pred['13C'][atom_idx]]
        d_exp_13C += [value] 

    # Convert calc list to shifted list
    d_exp_1H = np.array(d_exp_1H)
    d_calc_1H = np.array(d_calc_1H)
    d_exp_13C = np.array(d_exp_13C)
    d_calc_13C = np.array(d_calc_13C) 


    p_slope, p_intercept, _, _, _ = linregress(d_exp_1H, d_calc_1H)
    c_slope, c_intercept, _, _, _ = linregress(d_exp_13C, d_calc_13C)

    d_scaled_1H = (d_calc_1H - p_intercept)/p_slope
    d_scaled_13C = (d_calc_13C - c_intercept)/c_slope

    # d_scaled_1H = (d_calc_1H - reg_1H.intercept_)/reg_1H.coef_
    # d_scaled_13C = (d_calc_13C - reg_13C.intercept_)/reg_13C.coef_

    # Get probabilities from t distributions
    p_1H = t.logsf(abs(d_scaled_1H - d_exp_1H)/dp4_params['1H']['std'], dp4_params['1H']['nu'])
    p_13C = t.logsf(abs(d_scaled_13C - d_exp_13C)/dp4_params['13C']['std'], dp4_params['13C']['nu'])

    # Multiply all together and return
    return (np.sum(p_1H), np.sum(p_13C))