# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:40:10 2018

@author: Administrator
"""

#import pymc3 as pm
import scipy as sp
import numpy as np  
#import theano.tensor as tt

'''
#定义观测时间（各对象观测时间相同）
time_obseved=[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
time=np.array(time_obseved)
#print(time.shape)
#print(time)
#print(time[0,1])


#定义一个协方差矩阵（对应一个ρ）
def H_p(rh0):
    H_p=np.zeros((11,11))
    for i in range(0,11):
        for j in range(0,11):
            if j==i:
                H_p[i,j]=1.
            else:
                H_p[i,j]=np.power(rh0,np.abs(time[j]-time[i]))
    return H_p
     
rho=sp.stats.uniform.rvs(0,1)
H_p(rho)       
print(H_p(rho)[2,3])
'''
K=30
n=11
rho=np.ones((1,K))
time_obseved=[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
time=np.matrix(time_obseved)

#此处感觉要被自己蠢哭了✖✖✖
def H(rho):
    H=np.zeros((K,n,n))
    for k in range(0,K):
        #H[k]=np.zeros((11,11))
        for i in range(0,n):
            for j in range(0,n):
                if j==i:
                    H[k][i,j]=1.0
                else:
                    H[k][i,j]=np.power(rho[0,k],np.abs(time[0,j]-time[0,i]))        
    return H
        
  
rho=sp.stats.uniform.rvs(0,1,size=(1,K))
#print(rho)
print(H(rho))

   





