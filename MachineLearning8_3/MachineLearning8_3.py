# 推荐算法
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy import optimize as opt


data=loadmat("ex8_movieParams.mat")
data2=loadmat("ex8_movies.mat")
X=np.array(data["X"])
Theta=np.array(data["Theta"])
Y=np.array(data2["Y"])
R=np.array(data2["R"])
Params=np.concatenate((np.ravel(X),np.ravel(Theta)))


def Cost(Params,Y,R):       
    num_movies=Y.shape[0]
    num_users=Y.shape[1]
    X=np.array(np.reshape(Params[:num_movies*10],(num_movies,10)))
    Theta=np.array(np.reshape(Params[num_movies*10:],(num_users,10)))
    error=np.multiply(X@Theta.T-Y,R)
    squared_error=np.power(error,2)
    J=(1.0/2)*np.sum(squared_error)
    return J


def Gradient(Params,Y,R):
    num_movies=Y.shape[0]
    num_users=Y.shape[1]
    X=np.array(np.reshape(Params[:num_movies*10],(num_movies,10)))
    Theta=np.array(np.reshape(Params[num_movies*10:],(num_users,10)))
    error=np.multiply(X@Theta.T-Y,R)
    X_grad=error@Theta
    Theta_grad=error.T@X
    grad=np.concatenate((np.ravel(X_grad),np.ravel(Theta_grad)))
    return grad


def CostReg(Params,Y,R,lam):
    num_movies=Y.shape[0]
    num_users=Y.shape[1]
    X=np.array(np.reshape(Params[:num_movies*10],(num_movies,10)))
    Theta=np.array(np.reshape(Params[num_movies*10:],(num_users,10)))
    J=Cost(Params,Y,R)
    J+=(lam/2)*np.sum(np.power(X,2))
    J+=(lam/2)*np.sum(np.power(Theta,2))
    return J


def GradientReg(Params,Y,R,lam):
    num_movies=Y.shape[0]
    num_users=Y.shape[1]
    X=np.array(np.reshape(Params[:num_movies*10],(num_movies,10)))
    Theta=np.array(np.reshape(Params[num_movies*10:],(num_users,10)))
    grad=Gradient(Params,Y,R)
    grad_X=np.reshape(grad[:num_movies*10],(num_movies,10))
    grad_Theta=np.reshape(grad[num_movies*10:],(num_users,10))
    grad_X+=lam*X
    grad_Theta+=lam*Theta
    return np.concatenate((np.ravel(grad_X),np.ravel(grad_Theta)))


movie_list={}       #创建电影表(训练模型时不需要,但后续分析数据时可能用到)
with open("movie_ids.txt",encoding="gbk") as f:
    for line in f:
        tokens=line.split(" ")
        tokens[-1]=tokens[-1][:-1]      #去除'\n'
        movie_list[int(tokens[0])-1]=" ".join(tokens[1:])
        

YMean=np.zeros((Y.shape[0],1))      #归一化
Ynorm=np.zeros(Y.shape)
for i in range(Y.shape[0]):
    idx=np.where(R[i,:]==1)[0]
    YMean[i]=Y[i,idx].mean()
    Ynorm[i,idx]=Y[i,idx]-YMean[i]
    

fmin=opt.minimize(
                    fun=CostReg,
                    x0=Params,
                    args=(Ynorm,R,1),
                    method="CG",
                    jac=GradientReg,
                )


Xnew=np.array(np.reshape(fmin.x[:Y.shape[0]*10],(Y.shape[0],10)))
Thetanew=np.array(np.reshape(fmin.x[Y.shape[0]*10:],(Y.shape[1],10)))
Pre=Xnew@Thetanew.T+YMean       #预测结果


idx=np.where(R[0,:]==0)[0]
print(Pre[0,idx])       #预测未对第一部电影评分的人员的分数