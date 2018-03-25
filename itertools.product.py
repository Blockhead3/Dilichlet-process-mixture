# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 10:45:10 2018

@author: Cabbage
"""


import pymc3 as pm
import scipy as sp
import itertools

#笛卡尔积

def G0(sigam1,sigma2):
    g=[]
    theta_w=sp.stats.uniform.rvs(0,sigam1,2)#产生3个数值
    #print theta_w
    rho=sp.stats.uniform.rvs(0,sigma2,2)
    #print rho
    for i in itertools.product(theta_w,rho):
        g.append(list(i))
        
    return g
    
   
print G0(1,2)



