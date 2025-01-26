import pandas as pd
import numpy as np
from scipy.io import loadmat
import scipy.optimize as opt


data=loadmat("ex3data1.mat")
print(data)
data_y=data["y"]
y=np.zeros((10,data_y.shape[0]))        #1-9对应数字1-9,10对应数字0       
for i in range(5000):       
    y[data_y[i]-1,i]=1     
y=np.vstack([y[-1,:],y[:-1,:]])     #将第9行放置于第0行,由此矩阵y中1所在的行数(0-9行)分别对应数字0-9

    
data_x=data["X"]
X=np.array(data_x)
X=np.insert(X,obj=0,values=1,axis=1)        #在第0列插入全为1的一列


def sigmoid(z):     #sigmoid函数(s型函数)
    return 1/(1+np.exp(-z))


def CostFunctionReg(theta,X,y,lam):     #代价函数(正则化条件下)
    first=np.multiply(-y,np.log(sigmoid(X@theta)))
    second=np.multiply(1-y,np.log(1-sigmoid(X@theta)))
    return (1/X.shape[0])*np.sum(first-second)+lam/(2*X.shape[0])*np.sum(np.power(theta[1:],2))


def dJ(theta,X,y,lam):      #计算正则化条件下的的dJ/dθ
    d1=(1/X.shape[0])*X.T@(sigmoid(X@theta)-y)
    d2=(lam/X.shape[0])*theta
    d2[0]=0
    return d1+d2
    
    
def LogisticRegression(X,y,lam):
    theta=np.zeros(X.shape[1])      #opt.minimize函数要求theta必须是一维数组
    res=opt.minimize(
                    fun=CostFunctionReg,        #fun参数对应代价函数
                    x0=theta,       #优化的初始参数值
                    args=(X,y,lam),     #传递给fun和jac的额外参数
                    method="TNC",       #采用TNC算法
                    jac=dJ,     #jac对应dJ/dθ
                    options={"disp":True},      #显示优化过程
                    )
    final_theta=res.x
    return final_theta
"""
注意:
1.opt.minimize函数会自动调整alpha,因此不需要设置alpha
2.fun和jac会自动带入参数,因此只需填写函数名,不需填写函数参数
3.theta默认作为代价函数与dJ函数的第一个参数
4.返回的是一个一维数组形式的theta
"""


def Predict(X,theta):
    prob=sigmoid(X@theta)
    return (prob>=0.5).astype(int)      #(prob>=0.5)是一个0或1的bool类型,将其转换为int类型


t0=LogisticRegression(X,y[0],1)     #训练一个判断是否是数字0的t0
y_pre=Predict(X,t0)
accuracy=np.mean(y_pre==y[0])       #通过判断是否是数字0进行正确率评估
print(accuracy)


#基于上述的一维参数计算与应用方法,以下为k维参数计算与应用
t=np.array([LogisticRegression(X,y[i],1) for i in range(10)])       #将分别判断数字0-9的十个theta整合为t矩阵
prob_matrix=sigmoid(X@t.T)
y_prob=np.argmax(prob_matrix,axis=1)
data_y[data_y==10]=0
accuracy=np.mean(data_y.flatten()==y_prob)        #将data_y展平为一维数组并与y_prob进行比较形成bool数组,以此计算出正确率
print(accuracy)


#通过神经网络构建模型(神经网络相关参数已经由ex3weights.mat提供)
data1=loadmat("ex3data1.mat")
data2=loadmat("ex3weights.mat")
print(data1)
X=data1["X"]
y=data1["y"]
theta1=data2["Theta1"]
theta2=data2["Theta2"]
X=np.insert(X,obj=0,values=1,axis=1)
a1=X
z2=a1@theta1.T
a2=sigmoid(z2)
a2=np.insert(a2,obj=0,values=1,axis=1)
z3=a2@theta2.T
a3=sigmoid(z3)
print(a3)
print(a3.shape)
y_pred=np.argmax(a3,axis=1)+1      
accuracy=np.mean(y_pred==y.flatten())
print(accuracy)
#注意:此模型(神经网络)中10对应0，即[[0,0,0,0,0,0,0,0,0,1]].T表示0,而前面的模型以[[1,0,0,0,0,0,0,0,0,0]].T表示0,因此第96行操作较之前有所不同