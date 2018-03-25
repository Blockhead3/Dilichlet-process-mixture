# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 16:19:37 2018

@author: Cabbage

np.newaxis()
插入新维度

"""

import numpy as np
import pymc3 as pm
import scipy as sp

x=np.linspace(0,3,4)
print x
print np.shape(x)
#[ 0.  1.  2.  3.] 维度 4L
print x[:,np.newaxis]#行扩展的2维数组
print np.shape(x[:,np.newaxis])
'''
[[ 0.]
 [ 1.]
 [ 2.]
 [ 3.]]
--------4行1列(4L,1L)
'''

print x[np.newaxis,:]#列扩展的2维数组
print np.shape(x[np.newaxis,:])
#[[ 0.  1.  2.  3.]] ----- 1行4列(1L,4L)

print x[np.newaxis,np.newaxis,:]
print np.shape(x[np.newaxis,np.newaxis,:])
'''
[[[ 0.  1.  2.  3.]]] 3维数组 1行1列1个(1L,1L,4L)

'''

theta=sp.stats.norm.rvs(size=(5,3))
print theta
'''
[[-0.15104309 -1.24004623  0.14730115]
 [ 0.85819665 -1.17542383 -0.92647507]
 [-0.44875003 -0.48202521 -0.20536979]
 [-3.07004864  0.23249895  0.84763023]
 [ 0.20894809  0.62894981  1.18935644]]----5行3列(5L,3L)
'''
print theta[...,np.newaxis]
print np.shape(theta[...,np.newaxis])

'''
[[[-0.15104309]
  [-1.24004623]
  [ 0.14730115]]

 [[ 0.85819665]
  [-1.17542383]
  [-0.92647507]]

 [[-0.44875003]
  [-0.48202521]
  [-0.20536979]]

 [[-3.07004864]
  [ 0.23249895]
  [ 0.84763023]]

 [[ 0.20894809]
  [ 0.62894981]
  [ 1.18935644]]]-----维度(5L,3L,1L)
'''
print theta[...,np.newaxis][1]
'''
[[ 0.85819665]
  [-1.17542383]
  [-0.92647507]]
'''


#3维数组对应位置相乘
a=np.array([[1,0,1],
   [0,1,1],
   [2,0,1],
   [1,1,0]])
b=np.array([[2,1,2],
   [0,0,1],
   [0,1,0],
   [1,0,0]])
   
print a[...,np.newaxis]
'''
[[[1]
  [0]
  [1]]

 [[0]
  [1]
  [1]]

 [[2]
  [0]
  [1]]

 [[1]
  [1]
  [0]]]---(4L,3L,1L)
'''
print b[...,np.newaxis]
'''
[[[2]
  [1]
  [2]]

 [[0]
  [0]
  [1]]

 [[0]
  [1]
  [0]]

 [[1]
  [0]
  [0]]]---(4L,3L,1L)
'''
print a[...,np.newaxis]*b[...,np.newaxis]#对应位置相乘
'''
[[[2]
  [0]
  [2]]

 [[0]
  [0]
  [1]]

 [[0]
  [0]
  [0]]

 [[1]
  [0]
  [0]]]---(4L,3L,1L)
'''
print (a[...,np.newaxis]*b[...,np.newaxis]).sum(axis=1)#将每一行的2维（3行1列）数加起来
'''
[[4]
 [1]
 [0]
 [1]]-----(4L,1L)
'''
print (a[...,np.newaxis]*b[...,np.newaxis]).sum(axis=0)
'''
[[3]
 [0]
 [3]]---(3L,1L)
'''
