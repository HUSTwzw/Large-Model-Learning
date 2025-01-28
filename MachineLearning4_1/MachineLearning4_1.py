# 训练神经网络参数(利用反向传播)
import numpy as np
import pandas as pd
from scipy.io import loadmat
import scipy.optimize as opt


data=loadmat("ex4data1.mat")
print(data)
data_X=data["X"]
data_y=data["y"]
print(data_X)
print(data_y)
X=data_X
y=np.zeros((len(data_y),10))
for i in range(len(data_y)):
    y[i,data_y[i]-1]=1



def sigmoid(z):     #sigmoid函数(s型函数)
    return 1/(np.exp(-z)+1)


def Serialize(theta1,theta2):
    return np.concatenate((np.ravel(theta1),np.ravel(theta2)))


def Division(theta):        #将一维数组转换成矩阵theta1和theta2
    return theta[:25*401].reshape(25,401),theta[25*401:].reshape(10,26)     #在神经网络中,权重矩阵的每一行对应一个神经元的权重,权重的设计是"输出维度在前,输入维度在后"


def ForwardPropagation(theta,X):        #向前传播函数
    theta1,theta2=Division(theta)
    a1=np.insert(X,obj=0,values=1,axis=1)
    z2=a1@theta1.T
    a2=np.insert(sigmoid(z2),obj=0,values=1,axis=1)
    z3=a2@theta2.T
    h=sigmoid(z3)
    return a1,z2,a2,z3,h


def CostFunctionReg(theta,X,y,lam):     #代价函数(正则化版本)
    _,_,_,_,h=ForwardPropagation(theta,X)
    cost=np.sum(np.multiply(-y,np.log(h))-np.multiply((1-y),np.log(1-h)))/X.shape[0]
    theta1,theta2=Division(theta)
    reg=(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))*lam/(2*X.shape[0])
    return cost+reg
        

def dsigmoid(z):        #sigmoid函数的导数
    return np.multiply(sigmoid(z),1-sigmoid(z))    
        
        
def Gradient(theta,X,y):        #求梯度函数
    theta1,theta2=Division(theta)
    delta1=np.zeros(theta1.shape)
    delta2=np.zeros(theta2.shape)
    a1,z2,a2,z3,h=ForwardPropagation(theta,X)
    d3=h-y
    d2=np.multiply(d3@theta2[:,1:],dsigmoid(z2))
    delta2=(d3.T@a2)/X.shape[0]
    delta1=(d2.T@a1)/X.shape[0]
    return delta1,delta2


def GradientReg(theta,X,y,lam):     #求梯度函数(正则化版本)
    theta1,theta2=Division(theta)
    delta1,delta2=Gradient(theta,X,y)
    theta1[:,0]=0
    theta2[:,0]=0
    reg_1=(lam/X.shape[0])*theta1
    reg_2=(lam/X.shape[0])*theta2
    delta1+=reg_1
    delta2+=reg_2
    return Serialize(delta1,delta2)


def trainning(X,y,lam):     #训练参数theta的函数
    init_theta=np.random.uniform(-0.12,0.12,25*401+26*10)
    res=opt.minimize(fun=CostFunctionReg,
                     x0=init_theta,
                     args=(X,y,lam),
                     method="TNC",
                     jac=GradientReg,
                     options={"maxiter":200}       #设置迭代次数为200       #若不进行限制则迭代次数过大,导致耗费时间大
                    )
    final_theta=res.x
    return final_theta


def PredictAccuracy(theta,X,data_y):        #预测准确率
    _,_,_,_,h=ForwardPropagation(theta,X)
    y_pred=np.argmax(h,axis=1)+1
    accuracy=np.mean(y_pred==data_y.flatten())
    return accuracy


def CheckGradient(theta,X,y,epsilon):       #利用数学导数定义检验梯度函数的正确性(GradientReg函数)      #十分耗时
    theta_matrix=np.tile(theta,(len(theta),1))
    epsilon_matrix=np.identity(len(theta))*epsilon
    plus=theta_matrix+epsilon_matrix
    minus=theta_matrix-epsilon_matrix
    calculation=np.array([(CostFunctionReg(plus[i],X,y,1)-CostFunctionReg(minus[i],X,y,1))/(2*epsilon) for i in range(len(theta))])
    delta=GradientReg(theta,X,y,1)
    difference=np.linalg.norm(calculation-delta)/np.linalg.norm(calculation+delta)      #数值越小越准确
    print(difference)       


final_theta=trainning(X,y,1)
accuracy=PredictAccuracy(final_theta,X,data_y)
print(accuracy)
theta=np.random.uniform(-0.12,0.12,401*25+26*10)
CheckGradient(theta,X,y,0.0001)
# 注意:关于梯度计算,目前仍未掌握原理,同时计算步骤也不够熟练,需要后续学习进行补充