import tables
import os, sys
import mbgw


try:
    __root__ = mbgw.__path__[0] + '/../datafiles/auxiliary_data'
    os.listdir(__root__)
except:
    __root__ = '/datafiles/auxiliary_data'

class delayed_hdf(object):
    def __init__(self, fname):
        self.fname = fname
        self.initialized = False
        self.hfile = None
        
    def __getattr__(self, attr):
        if attr in ['fname','initialized', 'hfile']:
           return object.__getattr__(self, attr) 
        else:
            if not self.initialized:
                self.hfile = tables.openFile(self.fname)
                self.initialized = True
            return getattr(self.hfile.root, attr)
            

data_dict = {}
for fname in os.listdir(__root__):
    if fname[-4:]=='hdf5':
        try:
            # data_dict[fname[:-5]] = tables.openFile('%s/%s'%(__root__,fname)).root
            data_dict[fname[:-5]] = delayed_hdf('%s/%s'%(__root__,fname))
        except:
            cls, inst, tb = sys.exc_info()
            print 'Warning: unable to import auxiliary data. Error message: \n'+inst.message


locals().update(data_dict)
age_dist_model = age_dist_model.chain1.PyMCsamples
parameter_model = parameter_model.chain1.PyMCsamples
