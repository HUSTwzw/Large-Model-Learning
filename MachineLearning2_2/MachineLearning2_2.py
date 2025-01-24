# logistic回归与正则化(针对过拟合问题)
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


df=pd.read_csv("ex2data2.txt",header=None,names=["test1","test2","outcome"])
df_0=df[df["outcome"]==0]
df_1=df[df["outcome"]==1]
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(df_0["test1"],df_0["test2"],c="r",marker="x",label="failure")
plt.scatter(df_1["test1"],df_1["test2"],c="b",marker="o",label="success")
plt.legend()
plt.xlabel("test1")
plt.ylabel("test2")
plt.savefig("picture1.png")
plt.show()      #绘制原始数据图


y=np.array(df["outcome"]).reshape(-1,1)
df.drop("outcome",axis=1,inplace=True)       #删除"outcome"列
for i in range(3):
    for j in range(3):
        df["{}-{}".format(i,j)]=np.multiply(np.power(df["test1"],i),np.power(df["test2"],j))        #扩充数据,建立多项式
df.drop("test1",axis=1,inplace=True)        #删除"test1"列
df.drop("test2",axis=1,inplace=True)        #删除"test2"列


def sigmoid(z):     #sigmoid函数
    return 1/(1+np.exp(-z))


def CostFunction(X,y,theta,lam):        #代价函数
    first=np.multiply(-y,np.log(sigmoid(X@theta)))
    second=np.multiply(1-y,np.log(1-sigmoid(X@theta)))
    return (np.sum(first-second)/X.shape[0])+(lam/(2*X.shape[0]))*np.sum(np.power(theta[1:],2))


def GradientReg(X,y,theta,alpha,lam,iters):     #正则化函数
    for i in range(iters):
        error=sigmoid(X@theta)-y
        theta[0]=theta[0]-(alpha/X.shape[0])*np.sum(error)
        theta[1:]=(1-alpha*lam/X.shape[0])*theta[1:]-(alpha/X.shape[0])*(X.T[1:,:]@error)
    return theta    
        
        
def PredictOutcome(x,theta):        #预测结果
    if x@theta>=0.5:
        return 1
    else:
        return 0


def PredictAccuracy(X,y,theta):     #预测准确性
    count=0
    accuracy=0
    for i in range(X.shape[0]):
        if PredictOutcome(X[i,:].reshape(1,-1),theta)==y[i]:
           count+=1 
    accuracy=count/y.shape[0]
    return accuracy
    

X=np.array(df)
theta=np.zeros(9).reshape(-1,1)
theta=GradientReg(X,y,theta,0.01,1,100000)
print(theta)        #输出更新后的θ
print(CostFunction(X,y,theta,1))        #输出代价
print(PredictAccuracy(X,y,theta))       #输出准确性


u=np.linspace(df["0-1"].min()-0.5,df["0-1"].max()+0.5,100)      
v=np.linspace(df["1-0"].min()-0.5,df["1-0"].max()+0.5,100)
U,V=np.meshgrid(u,v)        #获取U矩阵(所有点的x坐标)与V矩阵(所有点的y坐标)


def GenerateFeature(test1,test2):       #用于生成特征函数
    feature=[]
    for i in range(3):
        for j in range(3):
            feature.append(np.power(test1,i)*np.power(test2,j))
    return np.array(feature).reshape(-1,9)      #返回x向量


Z=np.zeros(U.shape)
for i in range(U.shape[0]):
    for j in range(U.shape[1]):
        x=GenerateFeature(U[i,j],V[i,j])
        Z[i,j]=sigmoid(x@theta)     #对矩阵点通过拟合函数进行复制
        
        
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(df_0["test1"],df_0["test2"],c="r",marker="x",label="failure")
plt.scatter(df_1["test1"],df_1["test2"],c="b",marker="o",label="success")
plt.xlabel("test1")
plt.ylabel("test2")
plt.contour(U,V,Z,levels=[0.5],colors="green")      #绘制等值线
plt.legend()
plt.savefig("picture2.png")
plt.show()      