# 支持向量机
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat
import sklearn.svm
from sklearn import svm


# 线性SVM
data=loadmat("ex6data1.mat")
data_X=data["X"]
data_y=data["y"]
Data=pd.DataFrame(data_X,columns=["x1","x2"])
Data["y"]=data_y
positive=Data[Data["y"]==1]
negative=Data[Data["y"]==0]
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(positive["x1"],positive["x2"],c="b",label="positive")
plt.scatter(negative["x1"],negative["x2"],c="r",label="negative")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("picture1.png")
plt.show()      #展示初始数据


svc1=sklearn.svm.LinearSVC(C=1,loss="hinge")        #LinearSVC:线性支持向量分类器       hinge:损失函数类型(默认用于二分类问题)
svc1.fit(Data[["x1","x2"]],Data["y"])       #使用训练数据拟合模型    
score=svc1.score(Data[["x1","x2"]],Data["y"])       #计算模型准确率
print(score)
SVM1_confidence=svc1.decision_function(Data[["x1","x2"]])       #样本置信度(决策函数值):正值表示1,负值表示0,绝对值越大则置信度越高
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(Data["x1"],Data["x2"],c=SVM1_confidence,cmap="RdBu")        #设置为从蓝色渐变为红色
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SVM(C=1) Decision Confidence")
plt.savefig("picture2.png")
plt.show()      #绘制C=1训练出的置信度预测图


svc2=sklearn.svm.LinearSVC(C=100,loss="hinge")
svc2.fit(Data[["x1","x2"]],Data["y"])
score=svc2.score(Data[["x1","x2"]],Data["y"])
print(score)
SVM2_confidence=svc2.decision_function(Data[["x1","x2"]])
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(Data["x1"],Data["x2"],c=SVM2_confidence,cmap="RdBu")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("SVM(C=100) Decision Confidence")
plt.savefig("picture3.png")
plt.show()      #绘制C=100训练出的置信度预测图(C过大会导致过拟合)


# 非线性SVM
def GaussianKernel(x1,x2,sigma):        #高斯核函数(也称为RBF)(后续将不会自制此函数,只作为演示)
    return np.exp(-np.sum(np.power(x1-x2,2))/(2*(sigma**2)))


data2=loadmat("ex6data2.mat")
data2_X=data2["X"]
data2_y=data2["y"]
Data2=pd.DataFrame(data2_X,columns=["x1","x2"])
Data2["y"]=data2_y
positive2=Data2[Data2["y"]==1]
negative2=Data2[Data2["y"]==0]
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(positive2["x1"],positive2["x2"],c="b",label="positive")
plt.scatter(negative2["x1"],negative2["x2"],c="r",label="negative")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.savefig("picture4.png")
plt.show()      #展示初始数据


svc=svm.SVC(C=100,kernel="rbf",gamma=10,probability=True)       #rbf即高斯核函数的一种(RBF),gamma值控制高斯核函数的宽度(gamma越小决策边界越平滑),True表示计算不同分类的概率估计
svc.fit(Data2[["x1","x2"]],Data2["y"])
score=svc.score(Data2[["x1","x2"]],Data2["y"])
print(score)
prob=svc.predict_proba(Data2[["x1","x2"]])[:,0]     #计算出不同类的概率(第一列为0类的概率,第二列为1类的概率)
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(Data2["x1"],Data2["x2"],c=prob,cmap="Reds")
plt.xlabel("x1")
plt.ylabel("x2")
plt.savefig("picture5.png")
plt.show()


Data3=loadmat("ex6data3.mat")       #寻找最合适的C与gamma
training=pd.DataFrame(Data3["X"],columns=["x1","x2"])
training["y"]=Data3["y"]
cv=pd.DataFrame(Data3["Xval"],columns=["x1","x2"])
cv["y"]=Data3["yval"]
candidate=[0.01,0.03,0.1,0.3,1,3,10,30,100]
combination=[(C,gamma) for C in candidate for gamma in candidate]
evaluation=[]
for C,gamma in combination:
    svc=svm.SVC(C=C,kernel="rbf",gamma=gamma)
    svc.fit(training[["x1","x2"]],training["y"])
    score=svc.score(cv[["x1","x2"]],cv["y"])
    evaluation.append(score)
print(evaluation)
best_score=evaluation[np.argmax(evaluation)]
best_combination=combination[np.argmax(evaluation)]
print(best_score)
print(best_combination)