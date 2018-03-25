# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 10:41:59 2018

@author: Administrator
"""

from __future__ import division


from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import math
#from statsmodels.datasets import get_rdataset
from theano import tensor as tt

#生成数据
mu=np.linspace(0,0,num=9)

C1=np.zeros((9,9))
for i in range(0,9):
    for j in range(0,9):
        if i==j:
            C1[i,j]=10
        else:
            C1[i,j]=7
vals=np.random.multivariate_normal(mu,C1,size=50)

#模型
N = vals.shape[0]
K= 30

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


def cov(rho,K):
    Hrho_K=np.zeros((K,9,9))
    t=np.zeros((1,9))
    t[0,:]=np.arange(1,9+1,1)
    #print t
    rho=sp.stats.uniform.rvs(0,1,size=K)
    #print rho
    r=np.zeros((K,8))
    for k in range(K):
        
        for j in range(8):
            r[k,:]=pow(rho[k],abs(t[0,j+1]-t[0,j]))
        print r
        #Hrho[k]=np.ones((9,9))
        C=np.ones((9,9))
        for i in range(9):
             for j in range(9):
                if j==i:
                    C[i,j]==1.
                if (j>i) & (j<=8):
                    C[i,j]=r[k,:][i:j].prod(axis=0)
                 
                    C[j,i]=C[i,j]
        
        Hrho_K[k]=C
       
                
    return Hrho_K
    
    
'''
#模拟数据观测时间(假设各个体观测时间相同)    
t=np.zeros((1,9))
t[0,:]=np.arange(1,9+1,1)
#print t    
rho=sp.stats.uniform.rvs(0,1)
r=np.zeros((1,8))
for j in range(8):
    r[0,j]=pow(rho,abs(t[0,j+1] - t[0,j]))
#print r
Hrho=np.ones((9,9))
    #Hrho=np.vstack((Hrho_k))
for i in range(9):
    for j in range(9):
        if j==i:
            Hrho[i,j]==1.0
        if (j>i) & (j<=8):
            Hrho[i,j]=r[0,:][i:j].prod(axis=0)
        Hrho[j,i]=Hrho[i,j]
#print Hrho 
'''

with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    
    rho=pm.Uniform('rho',0,1,shape=K)
    sigma=pm.Uniform('sigma',0,1,shape=K)
    Hrho=pm.Deterministic('Hrho',cov(rho,K))
    
    dpm_cdf_components=pm.MvNormal('dpm_cdf_components',mu=np.zeros((K,9,1)),cov=sigma*Hrho)
    obs_cdfs=(w*dpm_cdf_components).sum(axis=1)
    obs=pm.Deterministic('obs',obs_cdfs,observed=vals)

with model:
    #start = find_MAP()
    step=pm.Metropolis()
    #step2 = pm.CategoricalGibbsMetropolis()
    trace = pm.sample(800,step=step)
    
    
    
          