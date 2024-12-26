import numpy
import random
print(numpy.nan==numpy.nan)     #nan不代表具体的数值，因此关系为!=而非==
print(numpy.nan!=numpy.nan)

arr=numpy.array([numpy.random.rand() for i in range(8)])
arr[[0,2,5]]=numpy.nan
print(arr)
arr=numpy.random.rand(1,8)
print(arr)
arr[:,[1,2,3]]=numpy.nan
print(arr)
amount=numpy.count_nonzero(arr!=arr)        #numpy.count_nonzero()返回非0个数，而arr!=arr只有在元素为nan时才成立，因此可以变相统计nan个数
print(amount)

a=numpy.isnan(arr)      #返回一个bool型列表，关于每一个元素是否是nan
print(type(a),a.dtype)
print(numpy.isnan(arr))
arr[numpy.isnan(arr)]=-1         #将arr中每一个nan元素替换为-1
print(arr)

arr=numpy.random.rand(3,4)
sum1=numpy.sum(arr)     #三种求和
sum2=numpy.sum(arr,axis=0)
sum3=numpy.sum(arr,axis=1)

Sum1=arr.sum()      #三种求和
Sum2=arr.sum(axis=0)
Sum3=arr.sum(axis=1)

middle1=numpy.median(arr)       #三种求中位数
middle2=numpy.median(arr,axis=0)
middle3=numpy.median(arr,axis=1)

Mean1=arr.mean()        #三种求平均值
Mean2=arr.mean(axis=0)
Mean3=arr.mean(axis=1)

ptp1=numpy.ptp(arr)     #三种求最大值与最小值的差
ptp2=numpy.ptp(arr,axis=0)
ptp3=numpy.ptp(arr,axis=1)

std1=arr.std()      #三种求标准差
std2=arr.std(axis=0)
std3=arr.std(axis=1)

print(arr)
print(sum1)
print(sum2)
print(sum3)
print(Sum1)
print(Sum2)
print(Sum3)
print(middle1)
print(middle2)
print(middle3)
print(Mean1)
print(Mean2)
print(Mean3)
print(ptp1)
print(ptp2)
print(ptp3)
print(std1)
print(std2)
print(std3)