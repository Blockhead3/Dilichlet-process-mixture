# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 09:26:33 2018

@author: Administrator
"""
import pymc3 as pm
import numpy as np
import theano.tensor as tt

import theano as T



init_mean=np.linspace(0,0,num=5)
init_sigma=np.linspace(1,1,num=5)
init_corr=np.ones((5))

data=np.random.multivariate_normal(init_mean,np.eye((5)),size=200)

def make_cov_matrix(sigma,C,module=tt):
    L=tt.diag(sigma)*C
    cov1 = tt.dot(L, L.T)
    return cov1
    
def multivariatenormal(init_mean, init_sigma, init_corr, suffix="", dist=False):
    if not isinstance(suffix, str):
        suffix = str(suffix)
    D = len(init_sigma)
    
    sigma = pm.Lognormal('sigma' + suffix, np.zeros(D, dtype=np.float), np.ones(D), shape=D, testval=init_sigma)
    nu = pm.Uniform('nu' + suffix, 0, 5)
    C_triu = pm.LKJCorr('C_triu' + suffix, nu, D, testval=init_corr)
    cov = pm.Deterministic('cov' + suffix, make_cov_matrix(sigma, C_triu, module=tt))
    
    mu = pm.MvNormal('mu' + suffix, 0, cov, shape=2, testval=init_mean)

    return pm.MvNormal.dist(mu, cov) if dist else pm.MvNormal('mvn' + suffix, mu, cov, shape=data.shape)

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining
'''
K = 3
with pm.Model() as model:
    #alpha = pm.Gamma('alpha', 1., 1.)
    #beta = pm.Beta('beta', 1, alpha, shape=K)
    #w = pm.Deterministic('w', stick_breaking(beta))
    w = pm.Dirichlet('w', np.ones(K))
    obs = pm.Mixture('obs', w, [multivariatenormal(init_mean, init_sigma, init_corr, k, dist=True) for k in range(K)], observed=data, shape=data.shape)

''' 
  
from pymc3.distributions.dist_math import bound
from pymc3.math import logsumexp

K = 3
with pm.Model() as model:
    #alpha = pm.Gamma('alpha', 1., 1.)
    #beta = pm.Beta('beta', 1, alpha, shape=K)
    #w = pm.Deterministic('w', stick_breaking(beta))
    w = pm.Dirichlet('w', np.ones(K))
    comp_dists = [multivariatenormal(init_mean, init_sigma, init_corr, k, dist=True) for k in range(K)]
    logpcomp = tt.stack([comp_dist.logp(T.shared(data)) for comp_dist in comp_dists], axis=1)
    pm.Potential('logp', logsumexp(tt.log(w) + logpcomp, axis=-1).sum())
    