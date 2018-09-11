# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 20:13:20 2018

@author: xiaochenchen
"""
from sklearn.decomposition import PCA  
import numpy as np  
from sklearn.preprocessing import StandardScaler  

x=np.array([[10001,2,55], [16020,4,11], [12008,6,33], [13131,8,22]])  
  
# feature normalization (feature scaling)  
X_scaler = StandardScaler()  
x = X_scaler.fit_transform(x)  
  
# PCA  
pca = PCA(n_components=0.9)# 保证降维后的数据保持90%的信息  
pca.fit(x)  
pca.transform(x)