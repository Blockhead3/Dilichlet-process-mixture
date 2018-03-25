# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 13:24:19 2018

@author: Cabbage
"""

import pymc3 as pm
import scipy as sp
import numpy as np  
import theano.tensor as tt

n=10
#模拟观测时间t(n*n_i)
t=np.zeros((n,9))
for k in range(n):
    t[k,:]=np.arange(k,9+k,1) #10个对象，每一个观测9次(等间隔)
#print t
    
#计算r_ij(n*n_i-1) j=1,2,,...,n_i-1
rho=[]
for k in range(n):
    rho.append(sp.stats.uniform.rvs(0,2))#采样ρ_k k=1,2,...n
#print rho

r=np.zeros((n,8))#每一行对应一个ρ_k，k=1,2,...,n
for k in range(n):
    for j in range(8):
        r[k,j]=pow(rho[k],abs(t[k,j+1] - t[k,j]))
#print r


#计算H(ρ_k)
#Hrho[]=np.ones((9,9))#写成zeros((9,9))
for k in range(n):
    Hrho_k=np.ones((9,9))
    #Hrho=np.vstack((Hrho_k))
for i in range(9):
    for j in range(9):
        if i==j:
            Hrho_k[i,j]=1.0
        if (j>i) & (j<=8):
            Hrho_k[i,j]=r[k,:][i:j].prod(axis=0)
        Hrho_k[j,i]=Hrho_k[i,j]
print Hrho_k
#print Hrho_k.shape


'''
n=5#5个观测对象
k=9#每个对象观测9次

#模拟观测时间
t=np.zeros((n,k))
for i in range(n):
    t[i,:]=np.arange(i,9+i,1) #n个对象，每一个观测k次(等间隔)
#print t

#定义r_ik
r=np.zeros((n,8))#每一行对应一个ρ_k，k=1,2,...,n
for k in range(n):
    for j in range(8):
        r[k,j]=pow(rho[k],abs(t[k,j+1] - t[k,j]))
#print r

#给定α的值
alpha=2
#σ_wi与ρ_i的
sigma=sp.stats.uniform.rvs(0.,1.,n)
rho=sp.stats.uniform.rvs(0.,1.,n)



r[i,:]=np.zeros((1,k-1))
    for j in range(k-1):
        r[j]=pow(rho[i],abs(t[i,j+1] - t[i,j]))
#print r
'''


#DPM例子，模拟了N=5组参数，每组参数确定一具体的DPM，且假定每个DPM是K=30类混合
N = 5
K = 30

alpha = 2
P0 = sp.stats.norm
f = lambda x, theta: sp.stats.norm.pdf(x, theta, 0.3)


beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]#第0列
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)#按行进行累乘
# w[:, 1:]--第1列到最后一列
# w[:, :-1]--最后1列之前的所以列
#这里的* 是对应位置元素相乘
theta = P0.rvs(size=(N, K))

dpm_pdf_components = f(x_plot[np.newaxis, np.newaxis, :], theta[..., np.newaxis])
dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=1)
# x_plot[np.newaxis, np.newaxis, :]将1维数据x_plot扩展成3维数组(1L,1L,200L)
# theta[..., np.newaxis]将2维theta扩展成3维(NL,KL,1L)
#x_plot分别对应每一个theta？？？？,有(NL,KL,1L)个正态分布密度函数
#w[..., np.newaxis]是(NL,KL,1L)维，对应位置相乘，按行累加，结果是二维(NL,1L)
#即产生N个DPM密度函数










  