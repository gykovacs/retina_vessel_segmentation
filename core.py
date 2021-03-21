import numpy as np
import pandas as pd
from scipy.optimize import linprog

from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor

import pulp as pl

def consistency_image_level(p, n, acc, sens, spec, eps):
    """
    Checking the consistency of image level figures
    
    Args:
        p (int): assumed number of positives
        n (int): assumed number of negatives
        acc (float): the observed accuracy score
        sens (float): the observed sensitivity score
        spec (float): the observed specificity score
        eps (float): the assumed +/- numerical uncertainty of the observed figures
    
    Returns:
        boolean: True if the observed scores are consistent with the assumed figures, False otherwise
    """
    term0= (n*(acc - spec) + p*(acc - sens) - 2*eps*(p+n)) <= 0
    term1= 0 <= (n*(acc - spec) + p*(acc - sens) + 2*eps*(p + n))
    term2= 0 >= p*(sens - eps - 1)
    term3= 0 <= p*(sens + eps)
    term4= 0 >= n*(spec - eps - 1)
    term5= 0 <= n*(spec + eps)
    return term0 & term1 & term2 & term3 & term4 & term5

def consistency_aggregated(p, n, acc, sens, spec, eps):
    """
    Checking the consistency of aggregated figures
    
    Args:
        p (np.array): vector of the assumed numbers of positives
        n (np.array): vector of the assumed numbers of negatives
        acc (float): the observed mean accuracy
        sens (float): the observed mean sensitivity
        spec (float): the observed mean specificity
        eps (float): the assumed +/- numerical uncertainty of the observed figures
    
    Returns:
        boolean: True if the observed scores are consistent with the assumed figures, False otherwise
    """
    num= len(n)
    
    c= np.hstack([np.repeat(1.0/num, num), np.repeat(0.0, num)])
    A_ub= np.array([np.hstack([1.0/(n*num), np.repeat(0.0, num)]),
                    np.hstack([-1.0/(n*num), -np.repeat(0.0, num)]),
                    np.hstack([np.repeat(0.0, num), 1.0/(p*num)]),
                    np.hstack([-np.repeat(0.0, num), -1.0/(p*num)]),
                    np.hstack([1.0/(n + p)/num, 1.0/(n + p)/num]),
                    np.hstack([-1.0/(n + p)/num, -1.0/(n + p)/num])])
    
    eps_sens= np.mean(1.0/p)
    eps_spec= np.mean(1.0/n)
    eps_acc= np.mean(1.0/(p + n))
    
    b_ub= np.array([spec + eps - eps_spec, -(spec - eps + eps_spec),
                    sens + eps - eps_sens, -(sens - eps + eps_sens),
                    acc + eps - eps_acc, -(acc - eps + eps_acc)])
    
    negpos_lower= np.hstack([np.repeat(0.0, len(n)), np.repeat(0.0, len(p))])
    negpos_upper= np.hstack([n, p])
    bounds= np.vstack([negpos_lower, negpos_upper]).T
    success= False
    try:
        res= linprog(c=-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        success= res['success']
    except Exception as e:
        print(e)
        success= False
        pass
    
    return success

def consistency_aggregated_integer_programming_mor(p, n, acc, sens, spec, eps):
    """
    Checking the consistency of aggregated figures, supposing mean of ratios calculation
    
    Args:
        p (np.array): vector of the assumed numbers of positives
        n (np.array): vector of the assumed numbers of negatives
        acc (float): the observed mean accuracy
        sens (float): the observed mean sensitivity
        spec (float): the observed mean specificity
        eps (float): the assumed +/- numerical uncertainty of the observed figures
    
    Returns:
        boolean: True if the observed scores are consistent with the assumed figures, False otherwise
    """
    prob= pl.LpProblem("feasibility")

    tps= [pl.LpVariable("tp" + str(i), 0, p[i], pl.LpInteger) for i in range(20)]
    tns= [pl.LpVariable("tn" + str(i), 0, n[i], pl.LpInteger) for i in range(20)]

    prob+= tps[0]

    prob+= sum([(1/20.0)*tps[i]*(1.0/p[i]) for i in range(20)]) <= sens + eps
    prob+= sum([(1/20.0)*(-1)*tps[i]*(1.0/p[i]) for i in range(20)]) <= -(sens - eps)
    prob+= sum([(1/20.0)*tns[i]*(1.0/n[i]) for i in range(20)]) <= spec + eps
    prob+= sum([(1/20.0)*(-1)*tns[i]*(1.0/n[i]) for i in range(20)]) <= -(spec - eps)
    prob+= sum([(1/20.0)*(tps[i] + tns[i])*(1.0/(p[i] + n[i])) for i in range(20)]) <= acc + eps
    prob+= sum([(1/20.0)*(-1)*(tps[i] + tns[i])*(1.0/(p[i] + n[i])) for i in range(20)]) <= -(acc - eps)

    return prob.solve() == 1

def consistency_aggregated_integer_programming_rom(p, n, acc, sens, spec, eps):
    """
    Checking the consistency of aggregated figures, supposing ratio of means calculation
    
    Args:
        p (np.array): vector of the assumed numbers of positives
        n (np.array): vector of the assumed numbers of negatives
        acc (float): the observed mean accuracy
        sens (float): the observed mean sensitivity
        spec (float): the observed mean specificity
        eps (float): the assumed +/- numerical uncertainty of the observed figures
    
    Returns:
        boolean: True if the observed scores are consistent with the assumed figures, False otherwise
    """
    prob= pl.LpProblem("feasibility")

    tps= [pl.LpVariable("tp" + str(i), 0, p[i], pl.LpInteger) for i in range(20)]
    tns= [pl.LpVariable("tn" + str(i), 0, n[i], pl.LpInteger) for i in range(20)]

    prob+= tps[0]

    prob+= sum([(1.0/(np.sum(p)))*tps[i] for i in range(20)]) <= sens + eps
    prob+= sum([(1.0/(np.sum(p)))*(-1)*tps[i] for i in range(20)]) <= -(sens - eps)
    prob+= sum([(1.0/(np.sum(n)))*tns[i] for i in range(20)]) <= spec + eps
    prob+= sum([(1.0/(np.sum(n)))*(-1)*tns[i] for i in range(20)]) <= -(spec - eps)
    prob+= sum([(1.0/(np.sum(p+n)))*(tps[i] + tns[i]) for i in range(20)]) <= acc + eps
    prob+= sum([(1.0/(np.sum(p+n)))*(-1)*(tps[i] + tns[i]) for i in range(20)]) <= -(acc - eps)

    return prob.solve() == 1

def score_range_image_level(p_fov, n_fov, p_diff, n_diff, acc_all, sens_all, spec_all, eps, n_diff_lower, score='acc'):
    """
    Score range reconstruction at the image level
    
    Args:
        p_fov (int): number of positives under the FoV
        n_fov (int): number of negatives under the FoV
        p_diff (int): number of positives outside the FoV
        n_diff (int): number of negatives outside the FoV
        acc_all (float): accuracy in the entire image
        sens_all (float): sensitivity in the entire image
        spec_all (float): specificity in the entire image
        eps (float): numerical uncertainty
        n_diff_lower (int): the minimum number of negatives outside the FoV
        score (str): score to compute: 'acc'/'sens'/'spec'
    
    Returns:
        float, float, float: the mid-score, the worst case minimum and best case maximum values
    """
    prob= pl.LpProblem("maximum", pl.LpMaximize)
    
    tp_fov= pl.LpVariable("tp_fov", 0, p_fov, pl.LpInteger)
    tn_fov= pl.LpVariable("tn_fov", 0, n_fov, pl.LpInteger)
    tp_diff= pl.LpVariable("tp_diff", 0, p_diff, pl.LpInteger)
    tn_diff= pl.LpVariable("tn_diff", n_diff_lower, n_diff, pl.LpInteger)
    
    prob+= (tp_fov + tp_diff)*(1.0/(p_fov + p_diff)) <= sens_all + eps
    prob+= (tp_fov + tp_diff)*(-1.0/(p_fov + p_diff)) <= -(sens_all - eps)
    
    prob+= (tn_fov + tn_diff)*(1.0/(n_fov + n_diff)) <= spec_all + eps
    prob+= (tn_fov + tn_diff)*(-1.0/(n_fov + n_diff)) <= -(spec_all - eps)
    
    prob+= (tp_fov + tp_diff + tn_fov + tn_diff)*(1.0/(p_fov + p_diff + n_fov + n_diff)) <= acc_all + eps
    prob+= (tp_fov + tp_diff + tn_fov + tn_diff)*(-1.0/(p_fov + p_diff + n_fov + n_diff)) <= -(acc_all - eps)
    
    if score == 'acc':
        prob+= (tp_fov + tn_fov)*(1.0/(p_fov + n_fov))
    elif score == 'spec':
        prob+= (tn_fov)*(1.0/n_fov)
    elif score == 'sens':
        prob+= (tp_fov)*(1.0/p_fov)
    
    prob.solve()
    
    score_max= pl.value(prob.objective)
    
    prob= pl.LpProblem("minimum", pl.LpMinimize)
    
    tp_fov= pl.LpVariable("tp_fov", 0, p_fov, pl.LpInteger)
    tn_fov= pl.LpVariable("tn_fov", 0, n_fov, pl.LpInteger)
    tp_diff= pl.LpVariable("tp_diff", 0, p_diff, pl.LpInteger)
    tn_diff= pl.LpVariable("tn_diff", n_diff_lower, n_diff, pl.LpInteger)
    
    prob+= (tp_fov + tp_diff)*(1.0/(p_fov + p_diff)) <= sens_all + eps
    prob+= (tp_fov + tp_diff)*(-1.0/(p_fov + p_diff)) <= -(sens_all - eps)
    
    prob+= (tn_fov + tn_diff)*(1.0/(n_fov + n_diff)) <= spec_all + eps
    prob+= (tn_fov + tn_diff)*(-1.0/(n_fov + n_diff)) <= -(spec_all - eps)
    
    prob+= (tp_fov + tp_diff + tn_fov + tn_diff)*(1.0/(p_fov + p_diff + n_fov + n_diff)) <= acc_all + eps
    prob+= (tp_fov + tp_diff + tn_fov + tn_diff)*(-1.0/(p_fov + p_diff + n_fov + n_diff)) <= -(acc_all - eps)
    
    if score == 'acc':
        prob+= (tp_fov + tn_fov)*(1.0/(p_fov + n_fov))
    elif score == 'spec':
        prob+= (tn_fov)*(1.0/n_fov)
    elif score == 'sens':
        prob+= (tp_fov)*(1.0/p_fov)
    
    prob.solve()
    
    score_min= pl.value(prob.objective)
    
    return (score_max + score_min)/2.0, (score_max - score_min)/2.0
    
def score_range_aggregated_mor(p_fov, n_fov, p_diff, n_diff, acc_all, sens_all, spec_all, eps, n_diff_lower, score='acc'):
    """
    Score range reconstruction for aggregated figures with the Mean-of-Ratios approach
    
    Args:
        p_fov (np.array): number of positives under the FoV
        n_fov (np.array): number of negatives under the FoV
        p_diff (np.array): number of positives outside the FoV
        n_diff (np.array): number of negatives outside the FoV
        acc_all (float): accuracy in the entire image
        sens_all (float): sensitivity in the entire image
        spec_all (float): specificity in the entire image
        eps (float): numerical uncertainty
        n_diff_lower (np.array): the minimum number of negatives outside the FoV
        score (str): score to compute: 'acc'/'sens'/'spec'
    
    Returns:
        float, float, float: the mid-score, the worst case minimum and best case maximum values
    """
    prob= pl.LpProblem("maximum", pl.LpMaximize)
    
    tp_fovs= [pl.LpVariable("tp_fov" + str(i), 0, p_fov[i], pl.LpInteger) for i in range(len(p_fov))]
    tn_fovs= [pl.LpVariable("tn_fov" + str(i), 0, n_fov[i], pl.LpInteger) for i in range(len(n_fov))]
    tp_diffs= [pl.LpVariable("tp_diff" + str(i), 0, p_diff[i], pl.LpInteger) for i in range(len(p_diff))]
    tn_diffs= [pl.LpVariable("tn_diff" + str(i), n_diff_lower[i], n_diff[i], pl.LpInteger) for i in range(len(n_diff))]
    
    sens_all_rom_plus= sum([(tp_fovs[i] + tp_diffs[i])*(1.0/(len(p_fov)*(p_fov[i] + p_diff[i]))) for i in range(len(p_fov))])
    spec_all_rom_plus= sum([(tn_fovs[i] + tn_diffs[i])*(1.0/(len(n_fov)*(n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    acc_all_rom_plus= sum([(tp_fovs[i] + tp_diffs[i] + tn_fovs[i] + tn_diffs[i])*(1.0/(len(n_fov)*(p_fov[i] + p_diff[i] + n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    sens_all_rom_minus= sum([(tp_fovs[i] + tp_diffs[i])*(-1.0/(len(p_fov)*(p_fov[i] + p_diff[i]))) for i in range(len(p_fov))])
    spec_all_rom_minus= sum([(tn_fovs[i] + tn_diffs[i])*(-1.0/(len(n_fov)*(n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    acc_all_rom_minus= sum([(tp_fovs[i] + tp_diffs[i] + tn_fovs[i] + tn_diffs[i])*(-1.0/(len(n_fov)*(p_fov[i] + p_diff[i] + n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    
    prob+= sens_all_rom_plus <= sens_all + eps
    prob+= sens_all_rom_minus <= -(sens_all - eps)
    
    prob+= spec_all_rom_plus <= spec_all + eps
    prob+= spec_all_rom_minus <= -(spec_all - eps)
    
    prob+= acc_all_rom_plus <= acc_all + eps
    prob+= acc_all_rom_minus <= -(acc_all - eps)
    
    if score == 'acc':
        prob+= sum([(tp_fovs[i] + tn_fovs[i])*(1.0/(len(p_fov)*(p_fov[i] + n_fov[i]))) for i in range(len(p_fov))])
    elif score == 'spec':
        prob+= sum([(tn_fovs[i])*(1.0/(len(p_fov)*(n_fov[i]))) for i in range(len(p_fov))])
    elif score == 'sens':
        prob+= sum([(tp_fovs[i])*(1.0/(len(p_fov)*(p_fov[i]))) for i in range(len(p_fov))])
    
    prob.solve()
    
    score_max= pl.value(prob.objective)
    
    prob= pl.LpProblem("minimum", pl.LpMinimize)
    
    tp_fovs= [pl.LpVariable("tp_fov" + str(i), 0, p_fov[i], pl.LpInteger) for i in range(len(p_fov))]
    tn_fovs= [pl.LpVariable("tn_fov" + str(i), 0, n_fov[i], pl.LpInteger) for i in range(len(n_fov))]
    tp_diffs= [pl.LpVariable("tp_diff" + str(i), 0, p_diff[i], pl.LpInteger) for i in range(len(p_diff))]
    tn_diffs= [pl.LpVariable("tn_diff" + str(i), n_diff_lower[i], n_diff[i], pl.LpInteger) for i in range(len(n_diff))]
    
    sens_all_plus= sum([(tp_fovs[i] + tp_diffs[i])*(1.0/(len(p_fov)*(p_fov[i] + p_diff[i]))) for i in range(len(p_fov))])
    spec_all_plus= sum([(tn_fovs[i] + tn_diffs[i])*(1.0/(len(n_fov)*(n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    acc_all_plus= sum([(tp_fovs[i] + tp_diffs[i] + tn_fovs[i] + tn_diffs[i])*(1.0/(len(n_fov)*(p_fov[i] + p_diff[i] + n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    sens_all_minus= sum([(tp_fovs[i] + tp_diffs[i])*(-1.0/(len(p_fov)*(p_fov[i] + p_diff[i]))) for i in range(len(p_fov))])
    spec_all_minus= sum([(tn_fovs[i] + tn_diffs[i])*(-1.0/(len(n_fov)*(n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    acc_all_minus= sum([(tp_fovs[i] + tp_diffs[i] + tn_fovs[i] + tn_diffs[i])*(-1.0/(len(n_fov)*(p_fov[i] + p_diff[i] + n_fov[i] + n_diff[i]))) for i in range(len(p_fov))])
    
    prob+= sens_all_plus <= sens_all + eps
    prob+= sens_all_minus <= -(sens_all - eps)
    
    prob+= spec_all_plus <= spec_all + eps
    prob+= spec_all_minus <= -(spec_all - eps)
    
    prob+= acc_all_plus <= acc_all + eps
    prob+= acc_all_minus <= -(acc_all - eps)
    
    if score == 'acc':
        prob+= sum([(tp_fovs[i] + tn_fovs[i])*(1.0/(len(p_fov)*(p_fov[i] + n_fov[i]))) for i in range(len(p_fov))])
    elif score == 'spec':
        prob+= sum([(tn_fovs[i])*(1.0/(len(p_fov)*(n_fov[i]))) for i in range(len(p_fov))])
    elif score == 'sens':
        prob+= sum([(tp_fovs[i])*(1.0/(len(p_fov)*(p_fov[i]))) for i in range(len(p_fov))])
    
    prob.solve()
    
    score_min= pl.value(prob.objective)

    return (score_min + score_max)/2.0, score_min, score_max

def score_range_aggregated_rom(p_fov, n_fov, p_diff, n_diff, acc_all, sens_all, spec_all, eps, n_diff_lower, score='acc'):
    """
    Score range reconstruction for aggregated figures with the Ratio-of-Means approach
    
    Args:
        p_fov (np.array): number of positives under the FoV
        n_fov (np.array): number of negatives under the FoV
        p_diff (np.array): number of positives outside the FoV
        n_diff (np.array): number of negatives outside the FoV
        acc_all (float): accuracy in the entire image
        sens_all (float): sensitivity in the entire image
        spec_all (float): specificity in the entire image
        eps (float): numerical uncertainty
        n_diff_lower (np.array): the minimum number of negatives outside the FoV
        score (str): score to compute: 'acc'/'sens'/'spec'
    
    Returns:
        float, float, float: the mid-score, the worst case minimum and best case maximum values
    """
    
    prob= pl.LpProblem("maximum", pl.LpMaximize)
    
    tp_fovs= [pl.LpVariable("tp_fov" + str(i), 0, p_fov[i], pl.LpInteger) for i in range(len(p_fov))]
    tn_fovs= [pl.LpVariable("tn_fov" + str(i), 0, n_fov[i], pl.LpInteger) for i in range(len(n_fov))]
    tp_diffs= [pl.LpVariable("tp_diff" + str(i), 0, p_diff[i], pl.LpInteger) for i in range(len(p_diff))]
    tn_diffs= [pl.LpVariable("tn_diff" + str(i), n_diff_lower[i], n_diff[i], pl.LpInteger) for i in range(len(n_diff))]
    
    sens_all_plus= sum([tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(1.0/(np.sum(p_fov + p_diff)))
    spec_all_plus= sum([tn_fovs[i] + tn_diffs[i] for i in range(len(p_fov))])*(1.0/(np.sum(n_fov + n_diff)))
    acc_all_plus= sum([tn_fovs[i] + tn_diffs[i] + tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(1.0/(np.sum(n_fov + n_diff + p_fov + p_diff)))
    
    sens_all_minus= sum([tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(-1.0/(np.sum(p_fov + p_diff)))
    spec_all_minus= sum([tn_fovs[i] + tn_diffs[i] for i in range(len(p_fov))])*(-1.0/(np.sum(n_fov + n_diff)))
    acc_all_minus= sum([tn_fovs[i] + tn_diffs[i] + tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(-1.0/(np.sum(n_fov + n_diff + p_fov + p_diff)))
    
    prob+= sens_all_plus <= sens_all + eps
    prob+= sens_all_minus <= -(sens_all - eps)
    
    prob+= spec_all_plus <= spec_all + eps
    prob+= spec_all_minus <= -(spec_all - eps)
    
    prob+= acc_all_plus <= acc_all + eps
    prob+= acc_all_minus <= -(acc_all - eps)
    
    if score == 'acc':
        prob+= sum([tp_fovs[i] + tn_fovs[i] for i in range(len(p_fov))])*(1.0/np.sum(p_fov + n_fov))
    elif score == 'spec':
        prob+= sum([tn_fovs[i] for i in range(len(p_fov))])*(1.0/np.sum(n_fov))
    elif score == 'sens':
        prob+= sum([tp_fovs[i] for i in range(len(p_fov))])*(1.0/np.sum(p_fov))
    
    prob.solve()
    
    score_max= pl.value(prob.objective)
    
    prob= pl.LpProblem("minimum", pl.LpMinimize)
    
    tp_fovs= [pl.LpVariable("tp_fov" + str(i), 0, p_fov[i], pl.LpInteger) for i in range(len(p_fov))]
    tn_fovs= [pl.LpVariable("tn_fov" + str(i), 0, n_fov[i], pl.LpInteger) for i in range(len(n_fov))]
    tp_diffs= [pl.LpVariable("tp_diff" + str(i), 0, p_diff[i], pl.LpInteger) for i in range(len(p_diff))]
    tn_diffs= [pl.LpVariable("tn_diff" + str(i), n_diff_lower[i], n_diff[i], pl.LpInteger) for i in range(len(n_diff))]
    
    sens_all_plus= sum([tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(1.0/(np.sum(p_fov + p_diff)))
    spec_all_plus= sum([tn_fovs[i] + tn_diffs[i] for i in range(len(p_fov))])*(1.0/(np.sum(n_fov + n_diff)))
    acc_all_plus= sum([tn_fovs[i] + tn_diffs[i] + tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(1.0/(np.sum(n_fov + n_diff + p_fov + p_diff)))
    
    sens_all_minus= sum([tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(-1.0/(np.sum(p_fov + p_diff)))
    spec_all_minus= sum([tn_fovs[i] + tn_diffs[i] for i in range(len(p_fov))])*(-1.0/(np.sum(n_fov + n_diff)))
    acc_all_minus= sum([tn_fovs[i] + tn_diffs[i] + tp_fovs[i] + tp_diffs[i] for i in range(len(p_fov))])*(-1.0/(np.sum(n_fov + n_diff + p_fov + p_diff)))
    
    prob+= sens_all_plus <= sens_all + eps
    prob+= sens_all_minus <= -(sens_all - eps)
    
    prob+= spec_all_plus <= spec_all + eps
    prob+= spec_all_minus <= -(spec_all - eps)
    
    prob+= acc_all_plus <= acc_all + eps
    prob+= acc_all_minus <= -(acc_all - eps)
    
    if score == 'acc':
        prob+= sum([tp_fovs[i] + tn_fovs[i] for i in range(len(p_fov))])*(1.0/np.sum(p_fov + n_fov))
    elif score == 'spec':
        prob+= sum([tn_fovs[i] for i in range(len(p_fov))])*(1.0/np.sum(n_fov))
    elif score == 'sens':
        prob+= sum([tp_fovs[i] for i in range(len(p_fov))])*(1.0/np.sum(p_fov))
    
    prob.solve()
    
    score_min= pl.value(prob.objective)

    return (score_min + score_max)/2.0, score_min, score_max

def score_range_aggregated(p_fov, n_fov, p_diff, n_diff, acc_all, sens_all, spec_all, eps, n_diff_lower, score='acc'):
    """
    Score range reconstruction for aggregated figures with both the Mean-of-Ratios and Ratio-of-Means approaches
    
    Args:
        p_fov (np.array): number of positives under the FoV
        n_fov (np.array): number of negatives under the FoV
        p_diff (np.array): number of positives outside the FoV
        n_diff (np.array): number of negatives outside the FoV
        acc_all (float): accuracy in the entire image
        sens_all (float): sensitivity in the entire image
        spec_all (float): specificity in the entire image
        eps (float): numerical uncertainty
        n_diff_lower (np.array): the minimum number of negatives outside the FoV
        score (str): score to compute: 'acc'/'sens'/'spec'
    
    Returns:
        float, float, float: the mid-score, the worst case minimum and best case maximum values
    """
    
    mean_rom, min_rom, max_rom= score_range_aggregated_rom(p_fov, n_fov, p_diff, n_diff, acc_all, sens_all, spec_all, eps, n_diff_lower, score)
    mean_mor, min_mor, max_mor= score_range_aggregated_mor(p_fov, n_fov, p_diff, n_diff, acc_all, sens_all, spec_all, eps, n_diff_lower, score)
    
    min_score= min([min_rom, min_mor])
    max_score= max([max_rom, max_mor])
    
    return (min_score + max_score)/2.0, (max_score - min_score)/2.0

def break_table(table, columns=2, index=False):
    """
    Technical function to break a long pandas dataframe into two columns

    Args:
        table (pd.DataFrame): pandas dataframe to break
        columns (int): the number of columns to break into
        index (boolean): whether to include the index or not
    
    Returns:
        pd.DataFrame: the pandas dataframe of multiple columns
    """
    n= len(table)
    n_items= [0] + [int(np.ceil(n/columns))]*columns
    for i in range(n - np.sum(n_items)):
        n_items[i]+= 1
    n_items= np.cumsum(n_items)
    pieces= []
    for i in range(len(n_items)-1):
        pieces.append(table.iloc[n_items[i]:n_items[i+1]].reset_index(drop=(not index)))
    
    return pd.concat(pieces, axis=1)

def set_column_spaces(latex, n_cols, col_space= 4, big_col_space=8):
    """
    Set the spaces between columns in Latex tables

    Args:
        latex (str): the Latex code of a tabular
        n_cols (int): the number of big columns (after breaking into multiple columns)
        col_space (int): points between columns
        big_col_space (int): points between big columns
    
    Returns:
        str: the adjusted Latex table code
    """
    pos= latex.find('\\begin{tabular}{')
    pos_tmp= (pos + len('\\begin{tabular}{'))
    first_half= latex[:pos_tmp]
    end= latex[pos_tmp:].find('}')
    second_half= latex[(pos_tmp + end):]
    columns= latex[pos_tmp:(pos_tmp + end)]
    total_cols= len(columns)


    result= ""
    for i in range(int(total_cols/n_cols)):
        for j in range(n_cols):
            result+= columns[i*n_cols + j]
            result+= '@{\\hspace{' + str(col_space) + 'pt}}'
        if i != int(total_cols/n_cols) - 1:
            result+= '@{\\hspace{' + str(big_col_space) + 'pt}}'
    
    return first_half + result + second_half

class PerformanceScoreAdjustment:
    """
    Fits regression models to aggregated and image level figures
    """
    def __init__(self, 
                 p, 
                 n, 
                 tp, 
                 tn, 
                 p_all, 
                 n_all, 
                 regressor=RandomForestRegressor(max_depth=6, random_state=5), 
                 std_mult=3.0, 
                 n_training= 10000, 
                 validator=RepeatedKFold(n_repeats=5, n_splits=5, random_state=5)):
        """
        Constructor of the object
        
        Args:
            p (np.array): the array of positives with FoV
            n (np.array): the array of negatives with FoV
            tp (np.array): the number of true positives with FoV
            tn (np.array): the number of true negatives with FoV
            p_all (np.array): the total number of positives (without FoV)
            n_all (np.array): the total number of negatives (without FoV)
            regressor (obj): the regressor object to be used
            std_mult (float): the multiplier of the standard deviation
            n_training (int): the number of training samples
            validator (obj): the validator object to be used
        """
        self.regressor= regressor
        self.n_training= n_training
        self.p, self.n, self.tp, self.tn= p, n, tp, tn
        self.p_all, self.n_all= p_all, n_all
        self.std_tp= np.std(tp)*std_mult
        self.std_tn= np.std(tn)*std_mult
        print(self.std_tp, self.std_tn)
        self.additional_p= p_all - p
        self.additional_n= n_all - n
        self.validator= validator
        self.r2_scores= {}
    
    def cross_validate(self, X, y):
        """
        Cross validates a particulare training set
        
        Args:
            X (np.array): the feature vectors
            y (np.array): the target values
        
        Returns:
            float: the r^2 score
        """
        y_pred, y_test= [], []

        for i, (training, test) in enumerate(self.validator.split(X, y)):
            y_pred.append(clone(self.regressor).fit(X[training], y[training]).predict(X[test]))
            y_test.append(y[test])

        return r2_score(np.hstack(y_test), np.hstack(y_pred))

    def fit_aggregated(self):
        """
        Fits a model to predict the aggregated FoV scores from aggregated no-FoV scores
        
        Returns:
            obj: the fitted object
        """
        X, y_acc, y_spec= [], [], []
        
        for _ in range(self.n_training):
            tp_tmp= np.clip(self.tp + np.round(np.random.normal(scale=self.std_tp, size=len(self.tp))), 0, self.p)
            tn_tmp= np.clip(self.tn + np.round(np.random.normal(scale=self.std_tn, size=len(self.tn))), 0, self.n)

            acc_fov= np.mean((tp_tmp + tn_tmp)/(self.p + self.n))
            acc_no_fov= np.round(np.mean((tp_tmp + tn_tmp + self.additional_p + self.additional_n)/(self.p + self.n + self.additional_p + self.additional_n)), 2)
            sens_no_fov= np.round(np.mean((tp_tmp + self.additional_p)/(self.p + self.additional_p)), 4)
            spec_fov= np.mean(tn_tmp/self.n)
            spec_no_fov= np.round(np.mean((tn_tmp + self.additional_n)/(self.n + self.additional_n)), 4)

            X.append([acc_no_fov, sens_no_fov, spec_no_fov])
            y_acc.append(acc_fov)
            y_spec.append(spec_fov)
        
        X= np.array(X)
        y_acc= np.array(y_acc)
        y_spec= np.array(y_spec)

        self.r2_scores['agg_acc']= self.cross_validate(X, y_acc)
        self.r2_scores['agg_spec']= self.cross_validate(X, y_spec)

        self.agg_acc_regressor= clone(self.regressor).fit(X, y_acc)
        self.agg_spec_regressor= clone(self.regressor).fit(X, y_spec)
        
        return self

    def fit_image_level(self):
        """
        Fits models to each image to predict the FoV scores from no-FoV scores
        
        Returns:
            obj: the fitted object
        """
        self.image_level_acc_regressor= {}
        self.image_level_spec_regressor= {}
        
        for i in range(len(self.p)):
            print('processing image %d' % i)
            
            if i == 0:
                print(self.tp[i], self.tn[i], self.p[i], self.n[i], self.p[i] + self.additional_p[i], self.n[i] + self.additional_n[i], self.std_tp, self.std_tn)

            tp_tmp= np.clip(self.tp[i] + np.round(np.random.normal(scale=self.std_tp, size=self.n_training)), 0, self.p[i])
            tn_tmp= np.clip(self.tn[i] + np.round(np.random.normal(scale=self.std_tn, size=self.n_training)), 0, self.n[i])

            acc_fov= (tp_tmp + tn_tmp)/(self.p[i] + self.n[i])
            acc_no_fov= np.round((tp_tmp + tn_tmp + self.additional_p[i] + self.additional_n[i])/(self.p[i] + self.n[i] + self.additional_p[i] + self.additional_n[i]), 4)
            sens_no_fov= np.round((tp_tmp + self.additional_p[i])/(self.p[i] + self.additional_p[i]), 4)
            spec_fov= tn_tmp/self.n[i]
            spec_no_fov= np.round((tn_tmp + self.additional_n[i])/(self.n[i] + self.additional_n[i]), 4)

            X= np.vstack([acc_no_fov, sens_no_fov, spec_no_fov]).T
            y_acc= acc_fov
            y_spec= spec_fov

            self.r2_scores['acc_' + str(i)]= self.cross_validate(X, y_acc)
            self.r2_scores['spec_' + str(i)]= self.cross_validate(X, y_spec)

            self.image_level_acc_regressor[i]= clone(self.regressor).fit(X, y_acc)
            self.image_level_spec_regressor[i]= clone(self.regressor).fit(X, y_spec)
            
        return self

    def fit(self):
        """
        Fit models for the aggregated and image level cases
        
        Returns:
            obj: the fitted object
        """
        self.fit_aggregated()
        self.fit_image_level()
        return self
    
    def predict_aggregated(self, X):
        """
        Predict aggregated FoV scores for aggregated no-FoV scores
        
        Args:
            X (np.array): the array containing the acc, sens and spec values in the rows
        
        Returns:
            float, float: the predicted aggregated accuracy and specificity scores
        """
        return self.agg_acc_regressor.predict(X), self.agg_spec_regressor.predict(X)
    
    def predict_image_level(self, X):
        """
        Predict aggregated FoV scores for image level no-FoV scores
        
        Args:
            X (np.array): the array containing the acc, sens and spec values in the rows
        
        Returns:
            float, float: the predicted aggregated accuracy and specificity scores
        """
        preds= np.array([[self.image_level_acc_regressor[i].predict(X[[i]])[0],
                            self.image_level_spec_regressor[i].predict(X[[i]])[0]] for i in range(len(X))])
        
        return np.mean(preds[:,0]), np.mean(preds[:,1])