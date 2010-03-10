"""
Must have the following in current working directory:
- CSE_Asia_and_Americas...hdf5 (pr-incidence trace)
- pr-falciparum (age-pr relationship trace)
- age-dist-falciparum (age distribution trace)
"""

disttol = 5./6378.
ttol = 1./12

import tables as tb
import numpy as np
import agecorr
from st_cov_fun import *
from pr_incidence import BurdenPredictor

a_pred = a_pred = np.hstack((np.arange(15), np.arange(15,75,5), [100]))
age_pr_file = tb.openFile('pr-falciparum')
age_dist_file = tb.openFile('age-dist-falciparum')

age_pr_trace = age_pr_file.root.chain0.PyMCsamples.cols
age_dist_trace = age_dist_file.root.chain0.PyMCsamples.cols
P_trace = age_pr_trace.P_pred[:]
S_trace = age_dist_trace.S_pred[:]
F_trace = age_pr_trace.F_pred[:]
age_pr_file.close()
age_dist_file.close()

two_ten_factors = agecorr.two_ten_factors(10000, P_trace, S_trace, F_trace)

from generic_mbg import FieldStepper, invlogit, histogram_reduce
from pymc import thread_partition_array
from pymc.gp import GPEvaluationGibbs
import pymc as pm
import mbgw
import os
root = os.path.split(mbgw.__file__)[0]
pm.gp.cov_funs.cov_utils.mod_search_path.append(root)

def check_data(input):
    pass
    
nugget_labels = {'sp_sub': 'V'}
obs_labels = {'sp_sub': 'eps_p_f'}

# Extra stuff for predictive ops.
n_facs = 1000
# postproc = invlogit

def vivax(sp_sub):
    cmin, cmax = thread_partition_array(sp_sub)
    out = sp_sub_b.copy('F')     
    ttf = two_ten_factors[np.random.randint(len(two_ten_factors))]
    
    pm.map_noreturn(vivax_postproc, [(out, sp_sub_0, sp_sub_v, p1, ttf, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out

def pr(eps_p_f, two_ten_facs=two_ten_factors):
    pr = eps_p_f.copy('F')
    pr = invlogit(pr) * two_ten_facs[np.random.randint(len(two_ten_facs))]
    return pr

N_year = 1./12
xplot = np.linspace(0.001,1,100)
xplot_aug = np.concatenate(([0],xplot))
def incidence(sp_sub, 
                two_ten_facs=two_ten_factors,
                p2b = BurdenPredictor('CSE_Asia_and_Americas_scale_0.6_model_exp.hdf5', N_year),
                N_year = N_year):
    pr = eps_p_f.copy('F')
    pr = invlogit(pr) * two_ten_facs[np.random.randint(len(two_ten_facs))]
    i = np.random.randint(len(p2b.f))
    mu = p2b.f[i](pr)
    
    # Uncomment and draw a negative binomial variate to get incidence over a finite time horizon.
    r = (p2b.r_int[i] + p2b.r_lin[i] * pr + p2b.r_quad[i] * pr**2)
    ar = pm.rgamma(beta=r/mu, alpha=r*N_year)

    return (1-np.exp(-ar))
    
map_postproc = [pr, incidence]
bins = np.array([0,.1,.5,1])

def binfn(arr, bins=bins):
    out = np.digitize(arr, bins)
    return out

bin_reduce = histogram_reduce(bins,binfn)

def bin_finalize(products, n, bins=bins, bin_reduce=bin_reduce):
    out = {}
    for i in xrange(len(bins)-1):
        out['p-class-%i-%i'%(bins[i]*100,bins[i+1]*100)] = products[bin_reduce][:,i+1].astype('float')/n
    out['most-likely-class'] = np.argmax(products[bin_reduce], axis=1)
    out['p-most-likely-class'] = np.max(products[bin_reduce], axis=1).astype('float') / n
    return out
        
extra_reduce_fns = [bin_reduce]    
extra_finalize = bin_finalize

metadata_keys = ['ti','fi','ui','with_stukel','chunk','disttol','ttol']

def mcmc_init(M):
    M.use_step_method(GPEvaluationGibbs, M.sp_sub, M.V, M.eps_p_f)

non_cov_columns = {'lo_age': 'int', 'up_age': 'int', 'pos': 'float', 'neg': 'float'}

from model import *