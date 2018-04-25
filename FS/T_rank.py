# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
t_value,p_value=stats.ttest_ind(x[np.where(y=='P')],x[np.where(y!='P')])
ind=np.argsort(p_value)
sorted_X=X.T[ind].T
