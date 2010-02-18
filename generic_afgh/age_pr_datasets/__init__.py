# Author: Anand Patil
# Date: 6 Feb 2009
# License: Creative Commons BY-NC-SA
####################################

from csv import reader
import os
import numpy as np
import mbgw
__root__ = mbgw.__path__[0]
fnames = os.listdir(__root__+'/age_pr_datasets')

excl_names = ['cotonou', 'cotedivoire', 'legundi', 'manhica', 'purworejo', 'sukabumi']

methods = {'angola': 'RDT',
'linzolo': 'Microscopy',
'eritrea': 'RDT',
'navrongo': 'Microscopy',
'orissa': 'Microscopy',
'chonyi': 'Microscopy',
'ngerenya': 'Microscopy',
'kericho': 'Microscopy',
'kisii.gucha':'RDT',
'kisii': 'RDT',
'wosera': 'Microscopy',
'dielmo': 'Microscopy',
'ndiop': 'Microscopy',
'somaliaS':	'RDT',
'somaliaC':'RDT',
'somaliaNE': 'RDT',
'saotome': 'Microscopy',
'mchenga':'Microscopy',
'namawala':'Microscopy',
'tak_m': 'Microscopy',
'tak_r': 'RDT',
# Thailand  Tak Province    Microscopy/RDT
'vanuatu'	: 'RDT'}

datasets = {}
for fname in fnames:
    
    if not fname.split('.')[-1]=='age_pr_datasets':
        continue
    
    if fname.replace('.age_pr_datasets','') in excl_names:
        continue

    a_lo = []
    a_hi = []
    N = []
    pos = []
    neg = []
    r = reader(file(__root__+'/age_pr_datasets/'+fname), delimiter=' ')
    r.next()
    
    for line in r:
        a_lo.append(float(line[1]))
        a_hi.append(float(line[2]))
        N.append(int(line[5]))
        pos.append(int(line[7]))
        neg.append(int(line[8]))
        
    record = np.rec.fromarrays([a_lo, a_hi, N, pos, neg], names=['a_lo', 'a_hi', 'N', 'pos', 'neg'])
    datasets[fname.replace('.age_pr_datasets','')] = record
    # print fname.replace('.age_pr_datasets',''), sum(N)
    
a = np.hstack((np.arange(15), np.arange(15,75,5)))

def find_age_bins(a_lo, a_hi):
    
    slices = []
    bin_ctrs = []

    for a_tup in zip(a_lo, a_hi):
        i_min = np.where(a<=a_tup[0])[0][-1]
        i_max = np.where(a>=a_tup[1])[0]
        if len(i_max)==0:
            i_max = len(a)-1
        else:
            i_max = i_max[0]
        slices.append(slice(i_min, i_max))
        
        
        local_a = a[i_min:i_max+1]
        bin_ctrs.append((a_tup[0] + a_tup[1])/2.)
        
    return slices, np.array(bin_ctrs)

age_bin_ctrs = {}
age_slices = {}
for place in datasets.iterkeys():
    age_slices[place], age_bin_ctrs[place] = find_age_bins(datasets[place].a_lo, datasets[place].a_hi)