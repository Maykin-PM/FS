# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 10:13:23 2018

@author: Maykin
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics,cross_validation
def Ent(x):
    x_value_list=set([x[i] for i in range(x.shape[0])])
    ent=0.0
    for x_value in x_value_list:
        p=float(x[x==x_value].shape[0])/x.shape[0]
        logp=np.log2(p)
        ent-=p*logp
    return ent
def C_ent(x,y):
    y_value_list=set([y[i] for i in range(y.shape[0])])
    c_ent=0.0
    for y_value in y_value_list:
        py=float(y[y==y_value].shape[0])/y.shape[0]
        subx=x[y==y_value]
        c_ent-=py*Ent(subx)
    return c_ent
def SU(x,y):
    ig_xy=Ent(x)-C_ent(x,y)
    su=2*ig_xy/(Ent(x)+Ent(y))
    return su
X=pd.read_csv("G:\\FS\\data\\Adenoma.csv")
Y=pd.read_csv("G:\\FS\\class\\Adenomaclass.csv")
Y[Y=='P']=1
Y[Y=='N']=0
x=np.array(X.iloc[:,1:])
y=np.array(Y.iloc[:,2]).reshape(x.shape[1],1)
y=y.astype(np.int32)
x=Normalizer().fit_transform(x)
su_list=[]
print(x[1,:].shape)
for i in range(x.shape[0]):
    su_list.append(SU(x[i,:].T,y[:,0]))
ind=np.argsort(su_list)    
x=x[ind]
su_list.sort()
mark=[0]
sub_x=[x[0,:]]
for j in range(1,x.shape[0]):
    flg=0
    print(j)
    for i in mark:
        su=SU(x[i,:],x[j,:])
        if(su>su_list[j]):
            flg=1
            break
    if(flg==0):
        sub_x.append(x[j,:])
        mark.append(j)
sub_x=np.array(sub_x)
    