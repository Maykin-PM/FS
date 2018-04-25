# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 11:13:41 2018

@author: Maykin
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
X=pd.read_csv("G:\\FS\\data\\Adenoma.csv")
Y=pd.read_csv("G:\\FS\\class\\Adenomaclass.csv")
x=np.array(X.iloc[:,1:])
x=Normalizer().fit_transform(x).T
y=np.array(Y.iloc[:,2])
w_rank=[]
for i in range(x.shape[1]):
    t_value,w_value=stats.ranksums(x[np.where(y=='P'),i],x[np.where(y!='P'),i])
    w_rank.append(w_value)
ind=np.argsort(w_rank)
sorted_X=X.T[ind].T
