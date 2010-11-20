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

a_pred = np.hstack((np.arange(15), np.arange(15,75,5), [100]))
age_pr_file = tb.openFile('pr-falciparum')
age_dist_file = tb.openFile('age-dist-falciparum')

age_pr_trace = age_pr_file.root.chain0.PyMCsamples.cols
age_dist_trace = age_dist_file.root.chain0.PyMCsamples.cols
P_trace = age_pr_trace.P_pred[:]
S_trace = age_dist_trace.S_pred[:]
F_trace = age_pr_trace.F_pred[:]
age_pr_file.close()
age_dist_file.close()

two_ten_factors = agecorr.age_corr_factors_from_limits(2, 10, 10000, a_pred, P_trace, S_trace, F_trace)

from generic_mbg import invlogit, histogram_reduce
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

non_cov_columns = {'lo_age': 'int', 'up_age': 'int', 'pos': 'float', 'neg': 'float'}

# Postprocessing stuff for mapping

def vivax(sp_sub):
    cmin, cmax = thread_partition_array(sp_sub)
    out = sp_sub_b.copy('F')     
    ttf = two_ten_factors[np.random.randint(len(two_ten_factors))]
    
    pm.map_noreturn(vivax_postproc, [(out, sp_sub_0, sp_sub_v, p1, ttf, cmin[i], cmax[i]) for i in xrange(len(cmax))])
    return out

def pr(sp_sub, two_ten_facs=two_ten_factors):
    pr = sp_sub.copy('F')
    pr = invlogit(pr) * two_ten_facs[np.random.randint(len(two_ten_facs))]
    return pr

N_year = 1./12
xplot = np.linspace(0.001,1,100)
xplot_aug = np.concatenate(([0],xplot))
def incidence(sp_sub, 
                two_ten_facs=two_ten_factors,
                p2b = BurdenPredictor('Africa+_scale_0.6_model_exp.hdf5', N_year),
                N_year = N_year):
    pr = sp_sub.copy('F')
    pr = invlogit(pr) * two_ten_facs[np.random.randint(len(two_ten_facs))]
    i = np.random.randint(len(p2b.f))
    mu = p2b.f[i](pr)
    
    # Uncomment and draw a negative binomial variate to get incidence over a finite time horizon.
    r = (p2b.r_int[i] + p2b.r_lin[i] * pr + p2b.r_quad[i] * pr**2)
    ar = pm.rgamma(beta=r/mu, alpha=r*N_year)

    out = (1-np.exp(-ar))
    out[np.where(out==0)]=1e-10
    out[np.where(out==1)]=1-(1e-10)
    return out

# params for naive risk mapping
r = .1/200
k = 1./4.2
ndraws = 100 # from the heterogenous biting parameter CAREFUL! this can bump up mapping time considerably if doing large maps
trip_duration = 30  # in days

def unexposed_risk_(f):
    def unexposed_risk(sp_sub, f=f, two_ten_facs=two_ten_factors):
        pr = sp_sub.copy('F')
        pr = invlogit(pr)*two_ten_facs[np.random.randint(len(two_ten_facs))]

        pr[np.where(pr==0)]=1e-10
        pr[np.where(pr==1)]=1-(1e-10)

        gams = pm.rgamma(1./k,1./k,size=ndraws)

        ur = pr*0
        fac = -r*k*((1-pr)**(-1./k)-1)*trip_duration*f
        for g in gams:
            ur += 1-np.exp(fac*g)
        ur /= len(gams)
        
        ur[np.where(ur==0)]=1e-10
        ur[np.where(ur==1)]=1-(1e-10)

        return ur
        
    unexposed_risk.__name__ = 'unexposed_risk_%f'%f
    return unexposed_risk
    
#map_postproc = [pr]+map(unexposed_risk_, [.001, .01, .1, 1.])
map_postproc=[pr, unexposed_risk_(1.)]

bins_list = [np.array([0,.1,.5,.75,1]),np.array([0,.05,.4,1]),np.array([0,.01,.05,.4,1])]

bin_reduce_list=[]
for i in xrange(len(bins_list)):

    def binfn(arr, bins=bins_list[i]):
        out = np.digitize(arr, bins)
        return out

    bin_reduce_list = bin_reduce_list+[histogram_reduce(bins_list[i],binfn)]

def bin_finalize(products, n, bins_list=bins_list, bin_reduce_list=bin_reduce_list):

    nbins=len(bins_list)
    out={}
    for j in xrange(nbins):
        bins=bins_list[j]
        bin_reduce=bin_reduce_list[j]

        for i in xrange(len(bins)-1):
            out['b%i-p-class-%i-%i'%(j,bins[i]*100,bins[i+1]*100)] = products[bin_reduce][:,i+1].astype('float')/n
            out['b%i-most-likely-class'%(j)] = np.argmax(products[bin_reduce], axis=1)
            out['b%i-p-most-likely-class'%(j)] = np.max(products[bin_reduce], axis=1).astype('float') / n

    return out

extra_reduce_fns = bin_reduce_list
extra_finalize = bin_finalize

metadata_keys = ['ti','fi','ui','with_stukel','chunk','disttol','ttol']

# Postprocessing stuff for validation

def pr(data, a_pred=a_pred, P_trace=P_trace, S_trace=S_trace, F_trace=F_trace):
    n = data.pos + data.neg
    obs = data.pos/n.astype('float')
    facs = agecorr.age_corr_factors(data.lo_age, data.up_age, 10000, a_pred, P_trace, S_trace, F_trace).T
    def f(sp_sub, facs=facs, n=n):
        p=pm.flib.invlogit(sp_sub)*facs[np.random.randint(10000)]
        return np.random.binomial(n.astype('int'),p).astype('float')/n
    return obs, n, f

validate_postproc=[pr]

def survey_likelihood(x, survey_plan, data, i):
    data_ = np.ones_like(x)*data[i]
    return pm.binomial_like(data_, survey_plan.n[i], pm.invlogit(x))

# Postprocessing stuff for survey evaluation

def simdata_postproc(sp_sub, survey_plan):
    p = pm.invlogit(sp_sub)
    n = survey_plan.n
    return pm.rbinomial(n, p)

# Initialize step methods
def mcmc_init(M):
    M.use_step_method(GPEvaluationGibbs, M.sp_sub, M.V, M.eps_p_f, ti=M.ti)
    def isscalar(s):
        return (s.dtype != np.dtype('object')) and (np.alen(s.value)==1) and (s not in M.eps_p_f_list)
    scalar_stochastics = filter(isscalar, M.stochastics)
    
    # The following two lines choose the 'AdaptiveMetropolis' step method (jumping strategy) for 
    # the scalar variables: nugget, scale, partial sill etc. It tries to update all of the variables
    # jointly, so each iteration takes much less time. 
    #
    # Comment them to accept the default, which is one-at-a-time Metropolis. This jumping strategy is
    # much slower, and known to be worse in many cases; but has been performing reliably for small
    # datasets.
    # 
    # The two parameters here, 'delay' and 'interval', control how the step method attempts to adapt its
    # jumping strategy. It waits for 'delay' iterations of one-at-a-time updates before it even tries to
    # start adapting. Subsequently, it tries to adapt every 'interval' iterations.
    # 
    # If any of the variables appear to not have reached their dense support before 'delay' iterations
    # have elapsed, 'delay' must be increased. However, it's good to have 'delay' be as small as possible
    # subject to that constraint.
    #
    # 'Interval' is the last parameter to fiddle; its effects can be hard to understand.
    M.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_stochastics, delay=50000, interval=500)
    #
    # The following line sets the size of jumps before the first adaptation. If the chain is 'flatlining'
    # before 'delay' iterations have elapsed, it should be decreased. However, it should be as large as
    # possible while still allowing many jumps to be accepted.
    
    M.step_method_dict[M.log_amp][0].proposal_sd *= .1


from model import *
