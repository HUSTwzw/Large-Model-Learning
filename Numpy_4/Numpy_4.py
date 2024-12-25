import numpy
arr=numpy.arange(24).reshape(4,6)
print(arr)
arr[:,[2,3]]=arr[:,[3,2]]       #交换两列
print(arr)
arr=numpy.arange(24).reshape(4,6)
print(arr)
arr[:,[1,2,3]]=arr[:,[2,3,1]]       #交换多列
print(arr)
arr=numpy.arange(24).reshape(4,6)
print(arr)
arr[[1,2],:]=arr[[2,1],:]       #交换两行
print(arr)
arr=numpy.arange(24).reshape(4,6)
print(arr)
arr[[0,1,2],:]=arr[[1,2,0],:]       #交换多行
print(arr)


arr1=numpy.array([0,1,2,3,4,5])
arr2=numpy.array([10,11,12,13,14,15])
arr_v=numpy.vstack((arr1,arr2))     #竖直拼接
print(arr_v)
arr_V=numpy.vstack((arr2,arr1))     #竖直拼接
print(arr_V)
arr_h=numpy.hstack((arr1,arr2))     #水平拼接
print(arr_h)
arr_H=numpy.hstack((arr2,arr1))     #水平拼接
print(arr_H)


arr=numpy.array([[3,1,2,5,0],[4,1,2,0,2],[3,8,2,2,1]])
print(numpy.argmax(arr,axis=0),type(numpy.argmax(arr,axis=0)))      #numpy.argmax(arr,axis=0)将确定每一列最大值所在的行数，并将其作为元素组成一个一维数组
print(numpy.argmax(arr,axis=1))     #numpy.argmax(arr,axis=1)将确定每一行最大值所在的列数，并将其作为元素组成一个一维数组
print(numpy.argmin(arr,axis=0))     #numpy.argmin(arr,axis=0)将确定每一列最小值所在的行数，并将其作为元素组成一个一维数组
print(numpy.argmin(arr,axis=1))     #numpy.argmin(arr,axis=1)将确定每一行最小值所在的列数，并将其作为元素组成一个一维数组


arr=numpy.zeros((3,4))      #arr为一个三行四列的全0矩阵             注意：若不使用astype，则生成的arr的元素为浮点型
print(arr,arr.dtype)


arr=numpy.ones((3,4)).astype("int64")       #arr为一个三行四列的全1矩阵             注意：若不使用astype，则生成的arr的元素为浮点型
print(arr,arr.dtype)
arr=numpy.eye(3)        #创建一个三阶单位矩阵


print(arr,arr.dtype)
arr=numpy.random.rand(2,3).astype("float64")        #此方法会生成一个给定形状的数组，数组每一个元素的值是在区间[0,1)内均匀分布的随机数
print(arr)


arr=numpy.random.randn(2,3)     #此方法会生成一个给定形状的数组，数组每一个元素的值是服从标准正态分布(均值为0,标准差为1)的随机数
print(arr)
arr=numpy.random.randint(1,10,(2,3))        #此方法会生成一个给定形状的数组，数组每一个元素的值是在区间[a,b)的整型随机数
print(arr,arr.dtype)
arr=numpy.random.uniform(1,10,(2,3))        #此方法会生成一个给定形状的数组，数组每一个元素的值是在区间[a,b)的浮点型随机数
print(arr)
numpy.random.seed(2)        #此方法用于在生成伪随机数时指定种子,指定相同的种子将会产生相同的随机数序列,这在调试代码的时候非常有用
t=numpy.random.randint(1,10,(2,3))
print(t)
numpy.random.seed(2)        ##注意：每次生成随机数时需要重新设置随机数种子
b=numpy.random.randint(1,10,(2,3))
print(b)


numpy.random.seed(2)
arr1=numpy.random.randint(1,10,(2,3))
numpy.random.seed(8)
arr2=numpy.random.randint(1,10,(2,3))
arr1=arr2       #arr1=arr2表示变量arr1与arr2相互影响(arr1的随arr2的变化而变化，arr2随arr1的变化而变化)
print(f"{arr1}\n{arr2}")
arr2[[0,1],[2,1]]=-1
print(f"{arr2}\n{arr1}")


numpy.random.seed(8)
arr2=numpy.random.randint(1,10,(2,3))
arr1=arr2[:]        #类似于=
print(f"{arr1}\n{arr2}")
arr1[[0,1],[1,2]]=-1
print(arr1)
print(arr2)


numpy.random.seed(8)
arr1=numpy.random.randint(1,10,(2,3))
arr2=arr1.copy()        #arr2=arr1.copy()表示将arr1的内容拷贝给arr2，arr1与arr2互不干扰
print(f"{arr1}\n{arr2}")
arr1[[0,1],[1,2]]=-1
print(arr1)
print(arr2)