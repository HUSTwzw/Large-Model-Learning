# 1.logistic回归(分类)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df=pd.read_csv("ex2data1.txt",header=None,names=["exam1","exam2","outcome"])
df_0=df[df["outcome"]==0]
df_1=df[df["outcome"]==1]
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(df_0["exam1"],df_0["exam2"],c="r",marker="x",label="failed")
plt.scatter(df_1["exam1"],df_1["exam2"],c="b",marker="o",label="success")
plt.legend()
plt.xlabel("exam1_score")
plt.ylabel("exam2_score")
plt.savefig("picture1.png")
plt.show()      #绘图观察分布情况


def sigmoid(z):     #sigmoid函数(s函数)
    return 1/(1+np.exp(-z))


def CostFunction(X,y,theta):        #logistic回归的代价函数
    first=np.multiply(-y,np.log(sigmoid(X@theta)))
    second=np.multiply((1-y),np.log(1-sigmoid(X@theta)))
    return np.sum(first-second)/(X.shape[0])


def GradientDecent(X,y,theta,alpha,iters):      #梯度下降函数(更新θ)
    cost=np.zeros(iters)
    for i in range(iters):
        error=sigmoid(X@theta)-y        #注意与MachineLearning1_2的差别
        theta=theta-(alpha/X.shape[0])*(X.T@error)      
        cost[i]=CostFunction(X,y,theta)
    return theta,cost       
    
    
def OutcomePredict(x,theta):        #判断输出结果
    result=sigmoid(x@theta)
    if result>=0.5:
        return 1
    elif result<0.5:
        return 0
    
    
def AccuracyPredict(X,y,theta):     #判断正确率(根据已有数据进行判断)
    count=0
    accuracy=0
    for i in range(X.shape[0]):
        if OutcomePredict(X[i,:].reshape(1,-1),theta)==y[i][0]:
            count+=1
        else:
            pass
    accuracy=count/(y.shape[0])
    return accuracy


df.insert(loc=0,column="theta0",value=1)
cols=df.shape[1]
df_X=df.iloc[:,0:cols-1]
df_y=df.iloc[:,cols-1]
X=np.array(df_X)        #提取X矩阵
y=np.array(df_y).reshape(-1,1)      #提取y向量
theta=np.array([0,0,0]).reshape(3,1)        #创建并初始化theta向量
theta,cost=GradientDecent(X,y,theta,0.003,500000)       
print(theta)        #输出拟合后的θ
print(cost)     #输出代价的变化情况
print(AccuracyPredict(X,y,theta))       #输出正确率
f=(-theta[0]/theta[2])-(theta[1]/theta[2]*X[:,1])       #建立直线解析式
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(df_0["exam1"],df_0["exam2"],c="r",marker="x",label="failed")
plt.scatter(df_1["exam1"],df_1["exam2"],c="b",marker="o",label="success")
plt.plot(X[:,1],f,c="g",label="division")
plt.legend()
plt.xlabel("exam1_score")
plt.ylabel("exam2_score")
plt.savefig("picture2.png")
plt.show()      #绘制分割图