# 优化机器学习模型(学习曲线)
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.optimize as opt
from matplotlib import pyplot as plt


def load_data():    
    data=sio.loadmat("ex5data1.mat")
    return map(np.ravel,[data["X"],data["y"],data["Xtest"],data["ytest"],data["Xval"],data["yval"]])
#训练集:X,y(训练模型)     交叉验证集:Xval,yval(确定正则化参数)      测试集:Xtest,ytest(评估模型性能)



X,y,Xtest,ytest,Xval,yval=load_data()       
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(X,y)
plt.xlim(-60,40)
plt.ylim(0,40)
plt.xlabel("water level")
plt.ylabel("flow")
plt.grid(alpha=0.8)
plt.savefig("picture1.png")
plt.show()      #绘制训练数据散点图(水库水位变化(X)与大坝流出水量(y))


X=np.insert(X.reshape(-1,1),obj=0,values=1,axis=1)      #插入截距项(全1的一列)
Xval=np.insert(Xval.reshape(-1,1),obj=0,values=1,axis=1)
Xtest=np.insert(Xtest.reshape(-1,1),obj=0,values=1,axis=1)


def Cost(theta,X,y):
    m=X.shape[0]
    return np.sum(np.power(X@theta-y,2))/(2*m)


def CostReg(theta,X,y,lam):
    m=X.shape[0]
    cost=Cost(theta,X,y)
    reg=np.sum(np.power(theta[1:],2))*(lam/(2*m))
    return cost+reg


def Gradient(theta,X,y):
    m=X.shape[0]
    return (X.T@(X@theta-y))/m


def GradientReg(theta,X,y,lam):
    m=X.shape[0]
    reg=theta.copy()
    reg[0]=0
    return Gradient(theta,X,y)+reg*(lam/m)


def LinearRegression(X,y,lam):
    theta=np.ones(X.shape[1])
    res=opt.minimize(
                        fun=CostReg,
                        x0=theta,
                        args=(X,y,lam),
                        method="TNC",
                        jac=GradientReg,
                        options={"disp":True}
                    )
    final_theta=res.x
    return final_theta


final_theta=LinearRegression(X,y,1)
print(final_theta)
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(X[:,1],y,c="r",label="training data")
plt.plot(X[:,1],X[:,1]*final_theta[1]+final_theta[0],c="g",label="prediction")
plt.ylim(-10,40)
plt.xlim(-60,40)
plt.xlabel("water level")
plt.ylabel("flow")
plt.legend()
plt.grid(alpha=0.8)
plt.savefig("picture2.png")
plt.show()      #展示简单拟合的直线


def LearningCurve(X,y,Xval,yval,lam):     #学习曲线
    training_cost=[]
    cv_cost=[]
    m=X.shape[0]
    for i in range(1,m+1):
        t0=LinearRegression(X[:i,:],y[:i],lam)
        tr=Cost(t0,X[:i,:],y[:i])
        cv=Cost(t0,Xval,yval)
        training_cost.append(tr)
        cv_cost.append(cv)
    return training_cost,cv_cost


training_cost,cv_cost=LearningCurve(X,y,Xval,yval,0)
plt.figure(figsize=(12,8),dpi=80)
plt.plot(np.arange(1,X.shape[0]+1),training_cost,c="g",label="train")
plt.plot(np.arange(1,X.shape[0]+1),cv_cost,c="r",label="cv")
plt.xlabel("m")
plt.legend()
plt.savefig("picture3.png")
plt.show()      #展示学习曲线(lam=1,未构建多项式，未归一化)


for i in range(2,9):        #建立一个八次项的多项式
    X=np.insert(X,obj=i,values=np.power(X[:,1],i),axis=1)
for i in range(2,9):
    Xval=np.insert(Xval,obj=i,values=np.power(Xval[:,1],i),axis=1)
for i in range(2,9):
    Xtest=np.insert(Xtest,obj=i,values=np.power(Xtest[:,1],i),axis=1)


for i in range(1,9):        #进行归一化
    X[:,i]=(X[:,i]-X[:,i].mean())/X[:,i].std()
for i in range(1,9):
    Xval[:,i]=(Xval[:,i]-Xval[:,i].mean())/Xval[:,i].std()
for i in range(1,9):
    Xtest[:,i]=(Xtest[:,i]-Xtest[:,i].mean())/Xtest[:,i].std()
    
    
training_cost,cv_cost=LearningCurve(X,y,Xval,yval,0)
plt.figure(figsize=(12,8),dpi=80)
plt.plot(np.arange(1,X.shape[0]+1),training_cost,c="g",label="train")
plt.plot(np.arange(1,X.shape[0]+1),cv_cost,c="r",label="cv")
plt.xlabel("m")
plt.legend()
plt.savefig("picture4.png")
plt.show()      #展示学习曲线(lam=0,已构建多项式,已归一化)


training_cost,cv_cost=LearningCurve(X,y,Xval,yval,1)
plt.figure(figsize=(12,8),dpi=80)
plt.plot(np.arange(1,X.shape[0]+1),training_cost,c="g",label="train")
plt.plot(np.arange(1,X.shape[0]+1),cv_cost,c="r",label="cv")
plt.xlabel("m")
plt.legend()
plt.savefig("picture5.png")
plt.show()      #展示学习曲线(lam=1,已构建多项式,已归一化)


training_cost,cv_cost=LearningCurve(X,y,Xval,yval,100)
plt.figure(figsize=(12,8),dpi=80)
plt.plot(np.arange(1,X.shape[0]+1),training_cost,c="g",label="train")
plt.plot(np.arange(1,X.shape[0]+1),cv_cost,c="r",label="cv")
plt.xlabel("m")
plt.legend()
plt.savefig("picture6.png")
plt.show()      #展示学习曲线(lam=100,已构建多项式,已归一化)


lam_candidate=[0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]
training_cost=[]
cv_cost=[]
for lam in lam_candidate:
    theta=LinearRegression(X,y,lam)
    training_cost.append(Cost(theta,X,y))
    cv_cost.append(Cost(theta,Xval,yval))
plt.figure(figsize=(12,8),dpi=80)
plt.plot(lam_candidate,training_cost,c="g",label="train")
plt.plot(lam_candidate,cv_cost,c="r",label="cv")
plt.xlabel("lam")
plt.legend()
plt.savefig("picture7.png")
plt.show()      #展示不同lam对应的学习曲线(lam=1时cv_cost最小)


test=[]
for lam in lam_candidate:
    theta=LinearRegression(X,y,lam)
    test.append(Cost(theta,Xtest,ytest))
plt.figure(figsize=(12,8),dpi=80)
plt.plot(lam_candidate,test,c="b",label="test")
plt.xlabel("lam")
plt.legend()
plt.savefig("picture8.png")
plt.show()      #展示不同lam对应的测试(lam=0.3时test最小)