# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 09:29:06 2017

@author: Administrator
"""

import numpy as np
import os
import math


mu=np.linspace(0,0,num=11)


C1=np.zeros((11,11))
for i in range(0,11):
    for j in range(0,11):
        if i==j:
            C1[i,j]=10
        else:
            C1[i,j]=7

C2=np.zeros((11,11))
for i in range(0,11):
    for j in range(0,11):
            C2[i,j]=10.*0.4**np.abs(i-j)
            
C3=np.zeros((11,11))
for i in range(0,11):
    for j in range(0,11):
            C3[i,j]=0.3*C1[i,j]+0.7*C2[i,j]
C4=np.zeros((11,11))
for i in range(0,11):
    for j in range(0,11):
            C4[i,j]=10./math.sqrt(1+abs(i-j))

            
dataSet1=np.random.multivariate_normal(mu,C1,size=300)
dataSet2=np.random.multivariate_normal(mu,C2,size=300)
dataSet3=np.random.multivariate_normal(mu,C3,size=300)
dataSet4=np.random.multivariate_normal(mu,C4,size=300)

f1=open('data1.txt','w')
for rows in dataSet1:
    f1.write(str(rows)+'\n').split()
f1.close()

f2=open('data2.txt','w')
for rows in dataSet2:
    f2.write(str(rows)+'\n')
f2.close()

f3=open('data3.txt','w')
for rows in dataSet3:
    f3.write(str(rows)+'\n')
f3.close()

f4=open('data4.txt','w')
for rows in dataSet4:
    f4.write(str(rows)+'\n')
f4.close()

 