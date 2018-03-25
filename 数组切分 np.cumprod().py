# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 11:05:37 2018

@author: Cabbage

a[:,1:]
a[:,:-1]
1-a[:,:-1]
的意思

"""
import numpy as np

'''
a=np.array([[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]])

print a            
print a[:,1:]
print a[:,:-1]
print 1-a[:,:-1]
'''

a=np.array([1,2,3])
print np.cumprod(a)
'''
[1 2 6]--1,1*2,1*2*3
'''

a=np.array([[1,2,3],[4,5,6]])
print np.cumprod(a,dtype=float)
'''
[   1.    2.    6.   24.  120.  720.]
1,1*2,1*2*3,1*2*3*4,...,1*2*3*4*...*6
'''

print np.cumprod(a,axis=0)#按列
'''
[[ 1  2  3] 第1列：1,1*4;
 [ 4 10 18]] 第3列：3,3*6
'''

print np.cumprod(a,axis=1)#按行
'''
[[  1   2   6]
 [  4  20 120]]
'''
a=np.array([[1,2,3],[4,5,6]])
b=np.array([[2,3,4],[1,2,2]])
print a[:,0]*b[:,-1]
print a[:,0]#返回第0列的一个数组[1,4],

