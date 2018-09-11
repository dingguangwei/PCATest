# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 21:24:36 2018

@author: xiaochenchen
"""

import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    x = np.matrix(((0.9,2.4,1.2,0.5,0.3,1.8,0.5,0.3,2.5,1.3),(1,2.6,1.7,0.7,0.7,1.4,0.6,0.6,2.6,1.1))).T
    print("=== 使用PCA降维，将x降至一维 ===")
    print("x=\n",x)
    
    print("1、先求每个属性样本值减去均值")
    (n,m) = x.shape
    xba = x-x.mean(axis=0)
    print("xba=",xba)
    
    print("2、求协方差矩阵")
    cov = np.cov(x.T)
    print("cov=\n",cov)
    
    print("3、求协方差矩阵的特征值和特征向量")
    w,v = np.linalg.eig(cov)
    print("特征值按照从大到小的顺序排列，特征向量矩阵是由特征值对应的单位特征向量组成的")
    print("特征值：  w=",w)
    print("特征向量：v=\n",v)
    
    print("4、求xba与最大特征值对应的特征向量的乘积")
    v_ = np.matrix(v[:,0])
    print("v_=",v_)
    y = xba*(v_.T)
    print("降为一维的数据y=",y)


