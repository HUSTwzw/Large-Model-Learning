# 异常检测:高斯分布
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy import stats


#读取并处理数据
data=loadmat("ex8data1.mat")      
print(data.keys())
data_X=data["X"]
data_Xval=data["Xval"]
data_yval=data["yval"]
X=np.array(data_X)
Xval=np.array(data_Xval)
yval=np.array(data_yval)        


#绘制原始数据图
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(X[:,0],X[:,1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("picture1.png")
plt.show()     


mu=X.mean(axis=0)
sigma=X.var(axis=0)
p=np.zeros((X.shape[0],X.shape[1]))
p[:,0]=stats.norm(mu[0],sigma[0]).pdf(X[:,0])       #计算训练集的概率密度
p[:,1]=stats.norm(mu[1],sigma[1]).pdf(X[:,1])       #计算训练集的概率密度
P=np.multiply(p[:,0],p[:,1])


pval=np.zeros((Xval.shape[0],Xval.shape[1]))
pval[:,0]=stats.norm(mu[0],sigma[0]).pdf(Xval[:,0])       #计算验证集的概率密度
pval[:,1]=stats.norm(mu[1],sigma[1]).pdf(Xval[:,1])       #计算验证集的概率密度
Pval=np.multiply(pval[:,0],pval[:,1])


def SelectEpsilon(Pval,yval):
    best_epsilon=0
    best_F1=0
    step=(Pval.max()-Pval.min())/10000
    for epsilon in np.arange(Pval.min(),Pval.max()+step,step):
        preds=Pval<epsilon
        tp=np.sum(np.logical_and(preds==1,yval.flatten()==1)).astype("float")     #逻辑"与"计算出true-positive的数量(需要注意:preds与yval的维数要保持一致(一维))
        fp=np.sum(np.logical_and(preds==1,yval.flatten()==0)).astype("float")     #逻辑"与"计算出false-positive的数量(需要注意:preds与yval的维数要保持一致(一维))
        fn=np.sum(np.logical_and(preds==0,yval.flatten()==1)).astype("float")     #逻辑"与"计算出false-negative的数量(需要注意:preds与yval的维数要保持一致(一维))
        precison=tp/(tp+fp)     #计算查准率
        recall=tp/(tp+fn)       #计算回归率
        F1=(2*precison*recall)/(precison+recall)        #计算F1-score
        if F1>best_F1:
            best_F1=F1
            best_epsilon=epsilon
    return best_F1,best_epsilon


best_F1,best_epsilon=SelectEpsilon(Pval,yval)
print(best_epsilon)


plt.figure(figsize=(12,8),dpi=80)
plt.scatter(X[P<best_epsilon,0],X[P<best_epsilon,1],s=80,c="r",marker="x",label="anomaly")
plt.scatter(X[:,0],X[:,1],s=20,c="g",marker="o",label="all data")
plt.legend()
plt.savefig("picture2.png")
plt.show()      #展示判断结果