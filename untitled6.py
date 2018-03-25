# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:11:02 2018

@author: Administrator
"""

import numpy as np
from theano import tensor as tt
import scipy as sp


H=np.array([[[1,2],
             [2,3]],
            [[4,4],
             [5,5]],
             [[1,1],
              [2,0]],
              [[7,8],
               [5,7]]])
print(H)
w=np.linspace(0,3,num=4)
print(w)
print(w[:,np.newaxis,np.newaxis])
w[:,np.newaxis,np.newaxis]**2

c=w[:,np.newaxis,np.newaxis]**2*H
print(c)

rho=sp.stats.uniform.rvs(0,1,size=10)
print(rho[2])