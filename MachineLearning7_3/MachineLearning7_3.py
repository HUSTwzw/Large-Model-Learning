# 降维:PCA
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt


data=loadmat("ex7data1.mat")
data=data["X"]
data=np.array(data)
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(data[:,0],data[:,1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("picture1.png")
plt.show()


def PCA(X):
    m=X.shape[0]
    X=(X-X.mean(axis=0))/X.std(axis=0)
    sigma=(1/m)*(X.T@X)
    U,S,V=np.linalg.svd(sigma)      #奇异值分解(分解特征值)
    return U,S,V


def ReduceFeature(X,U,k):
    U_reduced=U[:,:k]
    Z=X@U_reduced
    return Z


def RecoverFeature(Z,U,k):
    U_reduced=U[:,:k]
    return Z@U_reduced.T


U,S,V=PCA(data)       #S为奇异值向量
Z=ReduceFeature(data,U,1)
x=RecoverFeature(Z,U,1)
print(Z)
print(S/S.sum())        #展示不同特征所占总特征值的比重
cost=np.sum(np.power(data-x,2))/np.sum(np.power(data,2))        #计算偏差
print(1-cost)       #展示方差保留率


plt.figure(figsize=(12,8),dpi=80)
plt.scatter(data[:,0],data[:,1],c="r",label="origin data")
plt.scatter(x[:,0],x[:,1],c="g",label="recover data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("picture2.png")
plt.show()


# 降维应用实例(面部图片特征降维)
data2=loadmat("ex7faces.mat")
data2=data2["X"]
data2=np.array(data2)
print(data2.shape)
face1=data2[2000,:].reshape(32,32)
plt.imshow(face1)
plt.savefig("picture3.png")
plt.show()
U2,S2,V2=PCA(data2)
Z2=ReduceFeature(data2,U2,100)
x2=RecoverFeature(Z2,U2,100)
face2=np.reshape(x2[2000,:],(32,32))
plt.imshow(face2)
plt.savefig("picture4.png")     #可以发现还原时图像某些特征被模糊了(特征压缩过于严重)
plt.show()