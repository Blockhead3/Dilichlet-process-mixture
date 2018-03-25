# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:08:15 2018

@author: Cabbage
模型：
y_i=(μ+b_i)*e+w_i+ε_i
等价于分层模型：
y_i|μ,b_i,w_i,σ^2 ～N((μ+b_i)*e+w_i,σ^2I) → ni维
w_i|φ=(σ_wi^2,ρ_i) ～ N(0,σ_wi^2H(ρ_i)) → ni维
φ_1,φ_2,...,φ_n ～ G
              G ～ DP(G_0,M)
              b_i|ζ ～ N(0,ζ)
              σ ～ U(0，A)
              μ ～ N(μ_0,B)
              ζ ～ IG(b1,b2)
              

"""


from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp


alpha=2
f = lambda omega, sigma_k,rho_k: sp.stats.multivariate_normal.pdf(omega, sigma_k, rho_k)

beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)

sigma_k=pm.Uniform(0,1)
rho_k=pm.Uniform(0,1)

x_plot = np.linspace(-3, 3, 200)

dpm_pdf_components = f(x_plot[np.newaxis, np.newaxis, :], theta[..., np.newaxis])
dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=1)





with pm.Model as model:
    sigma=pm.Uniform('sigma',0,1)
    mu=pm.Normal('mu',0,0.1)
    xi=pm.InverseGamma(1,1)
    b_i=pm.Normal(0,xi)
    
    sigma_k=pm.Uniform(0,1)
    rho_k=pm.Uniform(0,1)
    y=pm.MvNormal('y',(mu+b_i)*e+omega,sigma^2*I)
    
    
    
    
    
    
    
    
    
    
    

