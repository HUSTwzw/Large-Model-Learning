# K-Means算法应用:图像压缩
import numpy as np
import pandas as pd
from scipy.io import loadmat
import random
from matplotlib import pyplot as plt


Data=loadmat("bird_small.mat")
Data=Data["A"]
Data=np.array(Data)
Data=Data/255
data=Data.reshape(Data.shape[0]*Data.shape[1],Data.shape[2])
print(data.shape)


def RandomCentroids(data,K):
    index=random.sample(range(data.shape[0]),K)
    init=data[index]
    return init


def FindClosestCentroids(data,centroids):
    m=data.shape[0]
    k=centroids.shape[0]
    c=np.zeros(m)
    for i in range(m):
        min_dist=100000000
        for j in range(k):
            dist=np.sum(np.power(data[i,:]-centroids[j,:],2))
            if dist<min_dist:
                min_dist=dist
                c[i]=j
    return c
    

def ComputeCentroids(data,c,K):
    m,n=data.shape
    new_centroids=np.zeros((K,n))
    for i in range(K):
        new_centroids[i,:]=np.mean(data[c==i],axis=0)
    return new_centroids


def RunKMeans(data,K,iters):
    init=RandomCentroids(data,K)
    centroids=init
    for i in range(iters):
        c=FindClosestCentroids(data,centroids)
        centroids=ComputeCentroids(data,c,K)
    return c,centroids


def CostKMeans(data,centroids,c):
    m=data.shape[0]
    K=centroids.shape[0]
    cost=0
    for i in range(K):
        cost+=np.sum(np.power(data[c==i]-centroids[i,:],2))
    cost/=m
    return cost


c,centroids=RunKMeans(data,16,20)       #简化为16种颜色
cost=CostKMeans(data,centroids,c)
for i in range(10):
    temp_c,temp_centroids=RunKMeans(data,16,20)
    temp_cost=CostKMeans(data,temp_centroids,temp_c)
    if temp_cost<cost:
        c=temp_c
        centroids=temp_centroids
        cost=temp_cost
Compression=centroids[c.astype("int"),:]        #将数据转化为对应的16种颜色
Compression=Compression.reshape(Data.shape[0],Data.shape[1],Data.shape[2])


plt.figure(figsize=(16,8))     
plt.imsave("picture1.png",Data)
plt.imshow(Data)
plt.show()
plt.figure(figsize=(16,8))
plt.imsave("picture2.png",Compression)
plt.imshow(Compression)
plt.show()