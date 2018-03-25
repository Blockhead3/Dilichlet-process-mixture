# -*- coding: utf-8 -*-
"""
Spyder Cabbage

狄利克雷过程
P~DP(a,P0)
P()=∑<i=1～∞>wi*δωi()
其中，wi是权重，ωi独立同分布于P0，δωi()是ωi的示性函数
且wi=βi*∏<j=1～i-1>(1-βj),βi独立同分布于Beta(1,a)
这里wi,ωi与βi的个数分别一样

This is a temporary script file.
"""

from __future__ import division

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp #SciPy 是基于Numpy构建的一个集成了多种数学算法和方便的函数的Python模块
import seaborn as sns#python数据可视化模块
from theano import tensor as tt
import pandas as pd #Pandas模块是Python用于数据导入及整理的模块

blue, *_ = sns.color_palette()

SEED = 5132290 # from random.org

np.random.seed(SEED)

N = 20
K = 30

alpha = 2.
P0 = sp.stats.norm

#从折棍子过程中抽取样本
beta = sp.stats.beta.rvs(1, alpha, size=(N, K))#生成N×K个 βi
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)

omega = P0.rvs(size=(N, K))

x_plot = np.linspace(-3, 3, 200)#-3到3之间等间隔取200个
#print(x_plot)
#有N组参数，每一组参数是一个K类的DP,如此便有N个这样的DP分布函数,每一行是对应200个x点的p(x)值，有200个值
sample_cdfs = (w[..., np.newaxis] * np.less.outer(omega, x_plot)).sum(axis=1)#DP分布函数
#print(sample_cdfs.shape)---(20,200)
#画分布函数
fig, ax = plt.subplots(figsize=(8, 6))#创建一个8*6点的图（图像的长和宽）

ax.plot(x_plot, sample_cdfs[0], c='red', alpha=0.75,
        label='DP sample CDFs');#第0行
ax.plot(x_plot, sample_cdfs[1:].T, c='gray', alpha=0.75);#第1行到最后一行,本来每一行有200个值，分别对应200个x_plot点，为何转置？
ax.plot(x_plot, P0.cdf(x_plot), c='k', label='Base CDF');

ax.set_title(r'$\alpha = {}$'.format(alpha));
ax.legend(loc=2);



'''
DPM
x_i|theta_i ~N(theta,0.3^2)
theta_1,...,theta_n ~ P
P ~DP(2,N(0,1))
即
f(x_i|P)=Σ<k=1~∞> w_k*f(x_i|theta~_K)
w_k=β_k*∏<j=1～i-1>(1-β_j),β_i独立同分布于Beta(1,a=2)
theta~_K独立同分布于N(0，1)
w_k与β_k个数相同
w_k与theta~_k个数相同，k代表混合项数

'''
N = 5
K = 30

alpha = 2
P0 = sp.stats.norm
f = lambda x, theta: sp.stats.norm.pdf(x, theta, 0.3)#x|theta 的条件密度

#折棍子过程
beta = sp.stats.beta.rvs(1, alpha, size=(N, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)

#采样theta
theta = P0.rvs(size=(N, K))

dpm_pdf_components = f(x_plot[np.newaxis, np.newaxis, :], theta[..., np.newaxis])
dpm_pdfs = (w[..., np.newaxis] * dpm_pdf_components).sum(axis=1)#混合密度(5,200),5个DPM
#x_plot[np.newaxis, np.newaxis, :]将x_plot扩展成3维数组（1,1,200），有1行，这一行是一个（1,200）二维数组
#theta[..., np.newaxis]将2维数组扩展成三维（5,30,1）有5行，每一行是一个（30,1）的二维数组
#对每个个x_i，分布获得其在30个参数theta_i下的30个密度函数，有5行这样的30参数，故每个x_i有5个密度函数的混合
#print(x_plot[np.newaxis, np.newaxis, :].shape)
#print(theta[..., np.newaxis])
#print(w[..., np.newaxis].shape)#(5,30,1)
#print(dpm_pdf_components.shape)#(5,30,200)
#print(dpm_pdfs.shape)#---(5,200)


fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_plot, dpm_pdfs.T, c='green');

ax.set_yticklabels([]);
plt.xlabel('x_plot')
plt.ylabel('5 noraml pdfs')

#对于某一个DPM，对其进行分解
fig, ax = plt.subplots(figsize=(8, 6))

ix = 1

ax.plot(x_plot, dpm_pdfs[ix], c='r', label='Density_ix');
ax.plot(x_plot, (w[..., np.newaxis] * dpm_pdf_components)[ix, 0],
        '--', c='g', label='Mixture components (weighted)');
ax.plot(x_plot, (w[..., np.newaxis] * dpm_pdf_components)[ix].T,
        '--', c='g');

ax.set_yticklabels([]);
ax.legend(loc=1);





