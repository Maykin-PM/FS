# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:43:00 2018

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
#SU：symmetrical uncertainty
X=pd.read_csv("G:\\FS\\data\\Adenoma.csv")
Y=pd.read_csv("G:\\FS\\class\\Adenomaclass.csv")
Y[Y=='P']=1
Y[Y=='N']=0
x=np.array(X.iloc[:,1:])
y=np.array(Y.iloc[:,2]).reshape(x.shape[1],1)
y=y.astype(np.int32)
x=Normalizer().fit_transform(x)
print(x[1,:].shape)
best=0.0
mark=0
for i in range(x.shape[0]):
    r_fc=SU(x[i,:],y[:,0])
    if(r_fc>best):
        best=r_fc
        mark=i
sub_x=[x[mark,:]]
x=np.delete(x,mark,axis=0)
num_iter=x.shape[0]-1
merit=best
r_fc=best
r_ff=0
k=1
stop=0
for j in range(num_iter):
    best=0.0
    print(j)
    for i in range(x.shape[0]):
        tr_ff=0.0
        for ex in sub_x:
            tr_ff+=SU(ex,x[i,:])
        tr_ff+=r_ff
        tr_fc=r_fc+SU(x[i,:],y[:,0])
        tmerit=tr_fc/np.sqrt(k+1+tr_ff)
        if(best<tmerit):
            best=tmerit
            mark=i
            m_rfc=tr_fc
            m_rff=tr_ff
    if(best>merit):
        k=k+1
        merit=best
        r_fc=m_rfc
        r_ff=m_rff
        sub_x.append(x[mark,:])
        stop=0
#stop...
    if(stop==3):
        break
    stop+=1
    x=np.delete(x,mark,axis=0)
sub_x=np.array(sub_x)
        
    
    