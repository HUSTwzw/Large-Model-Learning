import pandas 
import numpy
import string
pd1=pandas.Series(numpy.arange(10,20))      #创建带标签(index)一维数组
print(pd1,pd1.dtype,type(pd1))
pd2=pandas.Series(numpy.arange(2,22,2),index=[i for i in range(10)])        #创建带标签(index)一维数组，同时自定义标签
print(pd2,pd2.dtype,type(pd2))
pd3=pandas.Series(numpy.arange(10),index=[string.ascii_uppercase[i] for i in range(10)])        #创建带标签(index)一维数组，同时将大写字母作为标签      #string.ascii_uppercase是一个包含所有大写字母的字符串
print(pd3,pd3.dtype,type(pd3))
dict_stu={"name":"Wangziwen","age":18,"identity":"student","school":"HUST"}
pd4=pandas.Series(dict_stu)     #自动将字典的key作为标签，将alue作为数组元素
print(pd4,pd4.dtype,type(pd4))
print(pd4.index)        #输出数组的标签(index)
print(pd4.values)       #输出数组的值(values)


series1=pd1[pd1>15]     #Series的切片(切片用法与ndarray基本一致)
series2=pd1[[0,1,2,3,4,5]]
series3=pd1[1:4]
series4=pd1[1:5:2]
print(series1,series2,series3,series4)
print(pd3["B"],pd3[1])


pd5=pandas.Series(numpy.random.randn(5))
print(pd5)
print(pd5.argmax())     #ndarray可以使用的常见函数大多可以应用在Series上(argmax,argmin,clip,sum,mean,std)
print(pd5.argmin())
print(pd5.clip(0,0.5))
print(pd5.sum())
print(pd5.std())
print(pd5.mean())

pd6=pandas.read_csv("./601939.csv")
print(pd6)
print(type(pd6))
print(pd6.shape)
print(pd6.columns)
print(pd6.index)