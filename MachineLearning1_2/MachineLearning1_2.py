# 2.多变量线性回归
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


df=pd.read_csv("ex1data2.txt",header=None,names=["size","bedroom","profit"])
df["size"]=(df["size"]-df["size"].mean())/df["size"].std()      #特征归一化(Z-score标准化)
df["bedroom"]=(df["bedroom"]-df["bedroom"].mean())/df["bedroom"].std()
print(df)
df.insert(loc=0,value=1,column="theta0")
cols=df.shape[1]
df_X=df.iloc[:,0:cols-1]
df_y=df.iloc[:,cols-1]
X=np.array(df_X)
y=np.array(df_y).reshape(-1,1)
theta=np.zeros((X.shape[1],1))


def CostFunction(X,y,theta):
    cost_vector=np.power(X@theta-y,2)
    return np.sum(cost_vector)/(2*(X.shape[0]))


# 方法一:传统方法
def GradientDecent(X,y,theta,alpha,iters):
    parameters=int(theta.shape[0])
    cost=np.zeros(iters)
    for i in range(iters):
        error=X@theta-y
        theta=theta-alpha/(X.shape[0])*(X.T@error)      #直接利用矩阵乘积可以避免temp_theta(详见MachineLearning1_1)
        cost[i]=CostFunction(X,y,theta)
    return theta,cost


theta,cost=GradientDecent(X,y,theta,0.01,1000)
print(theta)
print(cost)


# 方法二:正规方程
theta2=np.linalg.inv(X.T@X)@X.T@y       #np.linalg.inv()是求逆矩阵的函数
print(theta)
print(theta2)
print(CostFunction(X,y,theta))
print(CostFunction(X,y,theta2))
# 注意:数据本身可能存在问题导致最后的输出结果不太好