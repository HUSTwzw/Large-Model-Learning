# 1.单变量线性回归
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


df=pd.read_csv("ex1data1.txt",header=None,names=["population","profit"])
print(df)
df.insert(loc=0,value=1,column="theta0")        #创建一列全为1的数组
print(df)
cols=df.shape[1]        #统计列数
df_X=df.iloc[:,0:cols-1]       #截取X所需的dataFrame  
df_y=df.iloc[:,cols-1]         #截取y所需的dataFrame
X=np.array(df_X)      #将X所需的dataFrame替换为数组(后续矩阵的计算要求必须是numpy的数组或矩阵)
y=np.array(df_y).reshape(-1,1)       #将y所需的dataFrame替换为y向量(后续矩阵的计算要求必须是numpy的数组或矩阵)
theta=np.array([0,0]).reshape(2,1)        #生成theta向量并进行初始化


def CostFunction(X,y,theta):         #代价函数
    cost_vector=np.power((X@theta)-y,2)        #(X*theta)-y构成一个误差向量
    return np.sum(cost_vector)/(2*(X.shape[0]))


def GradientDecent(X,y,theta,alpha,iters):      #梯度下降
    temp_theta=np.zeros(theta.shape)     #创建临时存储新theta的向量
    parameters=int(theta.shape[0])      #确定θ个数
    cost=np.zeros(iters)        
    for i in range(iters):
        error=(X@theta-y)       #error为误差矩阵
        for j in range(parameters):
            multi=np.multiply(error,X[:,j].reshape(-1,1))
            temp_theta[j,0]=theta[j,0]-(alpha/(X.shape[0]))*np.sum(multi)       #通过temp_theta储存新θ
        theta=temp_theta 
        cost[i]=CostFunction(X,y,theta)
    return theta,cost           


theta,cost=GradientDecent(X,y,theta,0.0001,100000)
print(theta)
print(cost)
x=np.linspace(df_X["population"].min(),df_X["population"].max(),100)
f=theta[0,0]+theta[1,0]*x       #设置函数
plt.figure(figsize=(12,8),dpi=80)
plt.plot(x,f,color="r",label="Prediction")       #绘制拟合直线
plt.scatter(df_X["population"],df_y,label="Origin data")        #绘制散点图
plt.xlabel("population")
plt.ylabel("profit")
plt.legend()
plt.savefig("picture1.png")
plt.show()      #展示散点图和拟合图


plt.figure(figsize=(12,8),dpi=80)
plt.plot(range(1,100001),cost)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.savefig("picture2.png")
plt.show()      #展示代价与迭代次数的关系图