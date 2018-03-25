# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 15:43:56 2018

@author: Administrator

"""
from  matplotlib  import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

K=30


M=2
e=np.ones((11,1))
I=np.eye(11)


omega=np.empty_like(e)#11,1
omega[0]=sp.stats.norm(loc=0.,scale=sigmaw)
for i in range(1,11):
    omega[i]=sp.stats.norm(loc=omega[i-1.]*rho,scale=sigmaw*sp.sqrt(1.-rho))
    
def pdf(omega):
    C=np.zeros((11,1))
    for i in range(0,11):
        C[i]=omega[i].pdf()
    return C[:,0].cumprod(axis=0)

    
with pm.Model() as model:
    mu=pm.Normal('mu',mu=0.,sd=0.1)
    xi=pm.InverseGamma('xi',1.,1.0)
    b=pm.Normal('b',mu=0.,sd=xi)
    sigam=pm.Uniform('sigma',0.,0.2)
    

    
    
    
    
    
    y=pm.MvNormal('y',mu=(mu+b)*e+omega,cov=sigma**2.*I)
    
    
    
