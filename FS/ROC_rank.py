# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:18:26 2018

@author: Maykin
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics,cross_validation
X=pd.read_csv("G:\\FS\\data\\Adenoma.csv")
Y=pd.read_csv("G:\\FS\\class\\Adenomaclass.csv")
Y[Y=='P']=1
Y[Y=='N']=0
x=np.array(X.iloc[:,1:])
y=np.array(Y.iloc[:,2]).reshape(x.shape[1],1).T
y=y.astype(np.int32)
x=Normalizer().fit_transform(x)
model = MLPClassifier(hidden_layer_sizes = (100,100,100,100,100,100),activation='relu', solver='adam', alpha=0.0001,max_iter = 10000)
roc_rank=[]
for i in range(x.shape[0]):
    print(i)
    ti_x=x[i,:]
    ti_x=ti_x.reshape(ti_x.shape[0],1)
    train_x,test_x,train_y,test_y = cross_validation.train_test_split(ti_x,y.T,test_size=0.4,random_state=27)
    model=model.fit(train_x,train_y)
    pre_y=model.predict_proba(test_x)
    pre_y=pre_y[:,1]
    ti_auc=metrics.roc_auc_score(test_y,pre_y)
    roc_rank.append(ti_auc)
ind=np.argsort(roc_rank)
sorted_X=X.T[ind].T
