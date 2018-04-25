import numpy as np
import pandas as pd
from scipy import stats
X=pd.read_csv('F:/FS4/data/Adenoma.csv')
Y=pd.read_csv('F:/FS4/class/Adenomaclass.csv')
X=np.array(X.iloc[:,1:]).T
Y=np.array(Y.iloc[:,2])
Y[Y=='P']=1
Y[Y=='N']=0
wrank=[]
for i in range(X.shape[1]):
	wv,p=stats.ranksums(X[:,i],Y)
	wrank.append(wv)
ind=np.array(wrank).reshape(X.shape[1],1).T
nx=np.row_stack((X,ind)).T
nx=nx[nx[:,36].argsort()]
nx=np.delete(nx,36,axis=1)
print(nx.shape)