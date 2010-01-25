# Copyright (C) 2010 Anand Patil, Pete Gething
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#from testmbgw import test
try:
    from testmbgw import test
except ImportError:
    pass

from model import *
import pymc as pm
import numpy as np
import os
from copy import copy
from correction_factors import age_corr_likelihoods, age_corr_factors, two_ten_factors
from scipy import interpolate as interp
from st_cov_fun import *
import time
import auxiliary_data
# import MAPData
import gc
# from get_covariates import extract_environment_to_hdf5
from tables import ObjectAtom
from generic_mbg import *


f_labels = ['eps_p_f']
fs_have_nugget = {'eps_p_f': True}
nugget_labels = {'eps_p_f': 'V'}
M_labels = {'eps_p_f': 'M'}
C_labels = {'eps_p_f': 'C'}
x_labels = {'eps_p_f': 'data_mesh'}
diags_safe = {'eps_p_f': True}

# Extra stuff for predictive ops.
n_facs = 1000
# postproc = invlogit
def map_postproc(eps_p_f, two_ten_facs=two_ten_factors(n_facs)):
    return invlogit(eps_p_f) * two_ten_facs[np.random.randint(n_facs)]
    
metadata_keys = ['ti','fi','ui','with_stukel','chunk','disttol','ttol']

def mcmc_init(M):
    M.use_step_method(FieldStepper, M.f, M.V, M.C_eval, M.M_eval, M.logp_mesh, M.eps_p_f, M.ti)

non_cov_columns = {'lo_age': 'int', 'up_age': 'int', 'pos': 'float', 'neg': 'float'}

#bins = np.array([0,.001,.01,.05,.1,.2,.4,1])
# 
#def binfn(arr, bins=bins):
#    out = np.digitize(arr, bins)
#    return out
# 
#bin_reduce = histogram_reduce(bins,binfn)
# 
#def bin_finalize(products, n, bins=bins, bin_reduce=bin_reduce):
#    out = {}
#    for i in xrange(len(bins)-1):
#        out['p-class-%i-%i'%(bins[i]*100,bins[i+1]*100)] = products[bin_reduce][:,i+1].astype('float')/n
#    out['most-likely-class'] = np.argmax(products[bin_reduce], axis=1)
#    out['p-most-likely-class'] = np.max(products[bin_reduce], axis=1).astype('float') / n
#    return out
#        
#extra_reduce_fns = [bin_reduce]
#extra_finalize = bin_finalize
