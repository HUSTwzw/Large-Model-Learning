import numpy
arr1=numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr1.transpose())      #转置的方式1
print(arr1)
print(arr1.T)        #装置的方式2
print(arr1)
print(arr1.swapaxes(1,0))        #转置的方式2
print(arr1)


arr2=numpy.arange(1,13).reshape(3,4)
arr2=arr2[2]        #选取特定一行
print(arr2,arr2.shape)
arr2=numpy.arange(24).reshape(12,2)
arr2=arr2[1:7:2]        #选取有规律的多行    
print(arr2,arr2.shape)
arr2=numpy.arange(24).reshape(8,3)
arr2=arr2[2:]       #选取连续多行
print(arr2)
arr2=numpy.arange(24).reshape(12,2)
arr2=arr2[[1,3,4,9,0]]      #选取特定的多行
print(arr2,arr2.shape)


arr3=numpy.arange(24).reshape(3,8)
arr3=arr3[:,2]     #选取特定的一列
print(arr3,arr3.shape)
arr3=numpy.arange(24).reshape(3,8)
arr3=arr3[:,2:]     #选取第二列至最后一列
print(arr3,arr3.shape)
arr3=numpy.arange(24).reshape(3,8)
arr3=arr3[:,2:6]        #选取连续多列
print(arr3,arr3.shape)
arr3=numpy.arange(24).reshape(3,8)
arr3=arr3[:,1:7:2]      #选取有规律的多列
print(arr3,arr3.shape)
arr3=numpy.arange(24).reshape(3,8)
arr3=arr3[:,[5,2,0]]        #选取特定的多列
print(arr3,arr3.shape)


arr4=numpy.arange(24).reshape(4,6)
arr4=arr4[2,4]      #选取特定的一个行列交汇点
print(arr4,arr4.shape)
arr4=numpy.arange(24).reshape(4,6)
arr4=arr4[1:3,2:5]      #选取多行多列(1-2行、2-4列的交汇)
print(arr4,arr4.shape)
arr4=numpy.arange(24).reshape(4,6)
arr4=arr4[[0,3,1],[5,2,0]]      #选取原数组特定的几个点
print(arr4,arr4.shape)


import random
arr4=numpy.arange(24).reshape(6,4)
arr4[2:5,0:2]=0     #将2-5行与0-2列的交汇处全部改为0
print(arr4)
arr4=numpy.array([random.randint(1,25) for i in range(24)],dtype=numpy.int64)
print(arr4)
print(arr4<20)      #关于数组的条件判断会返回bool型数组
arr5=arr4[arr4<16]      #由满足条件的元素重组数组
arr4[arr4>=16]=0        #根据布尔索引将>=16的元素都改为0
print(arr5)
print(arr4)
arr4=numpy.array([random.randint(1,25) for i in range(24)])
arr4=numpy.where(arr4<=16,5,2)      #numpy三元运算符(将arr4<=16的元素变为5，将arr4>16的元素变为2)
print(arr4)
arr4=numpy.arange(1,25).reshape(4,6)
arr4=arr4.clip(3,8)     #通过clip函数将<=3的元素变为3，将>=8的元素变为8
print(arr4)