# 无监督学习(K平均函数)
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.io import loadmat
import random


data=loadmat("ex7data2.mat")
data=pd.DataFrame(data["X"],columns=["x1","x2"])
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(data["x1"],data["x2"])
plt.ylim(0,6)
plt.xlim(-2,10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("picture1.png")
plt.show()


def RandomCentroids(X,K):
    m=X.shape[0]
    index=random.sample(range(m),K)
    X=np.array(X)
    init=X[index,:]
    return init


def FindClosestCentroids(X,centroids):
    m=X.shape[0]
    k=centroids.shape[0]
    c=np.zeros(m)
    X=np.array(X)
    for i in range(m):
        min_dist=1000000000
        for j in range(k):
            dist=np.sum(np.power(X[i,:]-centroids[j,:],2))
            if dist<min_dist:
                min_dist=dist
                c[i]=j
    return c


def ComputeCentroids(X,c,K):
    m,n=X.shape
    X=np.array(X)
    c=np.array(c)
    new_centroids=np.zeros((K,n))
    for i in range(K):
        new_centroids[i:]=np.mean(X[c==i],axis=0)
    return new_centroids


def RunKMeans(X,K,iters):
    m,n=X.shape
    init=RandomCentroids(X,K)
    centroids=init
    for i in range(iters):
        c=FindClosestCentroids(X,centroids)
        centroids=ComputeCentroids(X,c,K)
    return c,centroids
        
        
def CostKMeans(X,centroids,c,K):        #计算K的代价
    m=X.shape[0]
    cost=0
    X=np.array(X)
    centroids=np.array(centroids)
    for i in range(K):
        cost+=np.sum(np.power(X[c==i]-centroids[i],2))    
    cost/=m
    return cost


c,centroids=RunKMeans(data,3,20)        
cost=CostKMeans(data,centroids,c,3)
for i in range(10):
    c_temp,centroids_temp=RunKMeans(data,3,20)
    cost_temp=CostKMeans(data,centroids_temp,c_temp,3)
    if cost_temp<cost:
        c=c_temp
        centroids=centroids_temp
        cost=cost_temp      #通过不断追求最小代价防止陷入局部最优


category0=data[c==0]
category1=data[c==1]
category2=data[c==2]
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(category0["x1"],category0["x2"],c="r",label="category0")
plt.scatter(category1["x1"],category1["x2"],c="b",label="category1")
plt.scatter(category2["x1"],category2["x2"],c="g",label="category2")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("picture2.png")
plt.show()