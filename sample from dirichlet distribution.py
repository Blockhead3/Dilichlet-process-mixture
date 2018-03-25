# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 21:13:32 2017

@author: Cabbage
"""

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import pandas as pd
from theano.compile.ops import as_op

data_prior=np.random.dirichlet(0.1*np.ones(10))

data_sample=np.random.dirichlet(data_prior,size=100)
print data_sample.shape
data=data_sample
with pm.Model() as model:
    
    theta=pm.Dirichlet('theta',a=np.ones(data.shape[1]),shape=(data.shape[1]),testval=None)
    
    data_est=pm.DensityDist('data_est',lambda value:tt.log(tt.dot(theta,value.T)),observed=data)
    
    map_estimate=pm.find_MAP()
    trace=pm.sample(1500,tune=500,chains=1,njobs=1,progressbar=True,nuts_kwargs={'target_accept':0.85})
    

pm.traceplot(trace)

pm.summary(trace)

