# 异常检测:多元高斯分布
import numpy as np
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
plt.xlim(0,30)
plt.ylim(0,30)
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("picture1.png")
plt.show()


mu=X.mean(axis=0)
sigma=np.cov(X.T)       #计算协方差矩阵Σ
multi_normal=stats.multivariate_normal(mu,sigma)
grid_x,grid_y=np.mgrid[0:30:0.01,0:30:0.01]
pos=np.dstack((grid_x,grid_y))      #将两个矩阵融合为三维
pdf=multi_normal.pdf(pos)       #计算多元高斯分布


plt.figure(figsize=(12,8),dpi=80)
cont=plt.contourf(grid_x,grid_y,pdf,alpha=1.0,cmap="Reds",levels=10)        #绘制10层的等高线图       
plt.colorbar(cont,label="Probability Density")
plt.scatter(X[:,0],X[:,1],alpha=0.8,c="g",label="origin data")
plt.legend()
plt.savefig("picture2.png")
plt.show()


#筛选ε的函数
def SelectEpsilon(X,Xval,yval):
    mu=X.mean(axis=0)
    sigma=np.cov(X.T)
    multi_normal=stats.multivariate_normal(mu,sigma)
    pval=multi_normal.pdf(Xval)
    best_F1=0
    best_epsilon=0
    step=(pval.max()-pval.min())/10000
    for epsilion in np.arange(pval.min(),pval.max()+step,step):
        preds=pval<epsilion
        tp=np.sum(np.logical_and(preds==1,yval.flatten()==1)).astype("float")
        fp=np.sum(np.logical_and(preds==1,yval.flatten()==0)).astype("float")
        fn=np.sum(np.logical_and(preds==0,yval.flatten()==1)).astype("float")
        precison=tp/(tp+fp)
        recall=tp/(tp+fn)
        F1=(2*precison*recall)/(precison+recall)
        if F1>best_F1:
            best_F1=F1
            best_epsilon=epsilion
    return best_F1,best_epsilon
        

best_F1,best_epsilon=SelectEpsilon(X,Xval,yval)
print(best_epsilon)
mu=X.mean(axis=0)
sigma=np.cov(X.T)
multi_normal=stats.multivariate_normal(mu,sigma)
pval=multi_normal.pdf(X)


#绘制异常判断图
plt.figure(figsize=(12,8),dpi=80)
cont=plt.contourf(grid_x,grid_y,pdf,alpha=1.0,cmap="Reds",levels=10)
plt.colorbar(cont,label="Probability Density")
plt.scatter(X[:,0],X[:,1],s=20,c="g",marker="o",label="all data")
plt.scatter(X[pval<best_epsilon,0],X[pval<best_epsilon,1],s=80,c="b",marker="x",label="anomaly")
plt.legend()
plt.savefig("picture3.png")
plt.show()


#读取并处理高维数据
data2=loadmat("ex8data2.mat")       
X2=np.array(data2["X"])
Xval2=np.array(data2["Xval"])
yval2=np.array(data2["yval"])


best_F1_2,best_epsilon2=SelectEpsilon(X2,Xval2,yval2)
mu2=X2.mean(axis=0)
sigma2=np.cov(X2.T)
multi_normal2=stats.multivariate_normal(mu2,sigma2)
pval2=multi_normal2.pdf(X2)
Anomaly_X2=X2[pval2<best_epsilon2]
print(Anomaly_X2)       #展示所有异常数据