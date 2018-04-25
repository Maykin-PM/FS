import pandas  as pd
import numpy as np
import os
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier# 定义多层感知机分类算法
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import SelectFpr,chi2


# 导入数据集
path_X = os.path.join(os.getcwd(),'Adenoma.csv')# 获得其特征向量
X = pd.read_csv(path_X,skiprows = [0])
#除去第一列
X.drop([X.columns[0]], axis=1,inplace=True)
X = X.T
path_y = os.path.join(os.getcwd(),'Adenomaclass.csv')# 获得样本label
y = pd.read_csv(path_y,usecols=['Class'])
# #把 P/N 转换成1/0
y.rename(index = str, columns={'Class':0}, inplace = True)


for i in range(y.shape[0]):
	if y.iloc[i,0] == 'P':#iloc 需要用数字作为索引，loc需要用字符作为索引
		y.iloc[i,0] = 1
	else:
		y.iloc[i,0] = 0

y = np.ravel(y)#change a matrix into 1d array

#归一化，返回值为归一化后的数据
X = Normalizer().fit_transform(X)
"""
三种wrapper
"""

#用的是Lasso
# clf = LassoCV()
# clf = clf.fit(X,y)
# model = SelectFromModel(clf,prefit = True)
# X_new = model.transform(X)

# #用的是Ridge
# clf = RidgeCV()
# clf = clf.fit(X,y)
# model = SelectFromModel(clf, prefit = True)
# X_new = model.transform(X)

#随机森林
rfc = RandomForestClassifier(n_jobs = -1)
rfc.fit(X,y)
model = SelectFromModel(rfc, prefit = True)
X_new = model.transform(X)

"""

参数
---
    hidden_layer_sizes: 元组
    activation：激活函数
    solver ：优化算法{‘lbfgs’, ‘sgd’, ‘adam’}
    alpha：L2惩罚(正则化项)参数。
"""
model = MLPClassifier(hidden_layer_sizes = (200,100,100,100,100,100),activation='relu', solver='adam', alpha=0.0001,max_iter = 10000)

acc = cross_val_score(model, X_new, y,scoring = "accuracy", cv=10)
#f_score = cross_val_score(model, X, y,scoring = "f1", cv=10,pos_label = "P")
print(acc)
#print(f_score)
"""参数
---
    model：拟合数据的模型
    cv ： k-fold
    scoring: 打分参数-‘accuracy’、‘f1’、‘precision’、‘recall’ 、‘roc_auc’、'neg_log_loss'等等
"""