import numpy
arr=numpy.arange(1,25)      #创建一维数组
print(arr,arr.shape)        #查看数组的形状(维数)
arr=arr.reshape(3,8)        #改变数组的形状：将一维数组转化为(三行八列的)二维数组的形状
print(arr,arr.shape)
arr=numpy.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print(arr.shape)
arr=arr.reshape(5,3)
print(arr,arr.shape)
arr=numpy.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12]]])            #创建三维数组
print(arr,arr.shape)
arr=numpy.array([i for i in range(1,13)]+[i for i in range(-1,-13,-1)],dtype=numpy.int64)
arr=arr.reshape(2,3,4)      #将一维数组转化为三维数组
print(arr,arr.shape,arr.dtype,type(arr))
arr=arr.reshape(24)     #将三维数组变为一维数组的方式1
print(arr,arr.shape,type(arr),arr.dtype)        
arr=numpy.array([[1,2,3,4],[5,6,7,8]])      #创建二维数组
arr=arr.flatten()       #将数组展平(将数组转化为一维数组的方式2)
print(arr,arr.shape,type(arr),arr.dtype)
arr+=2      #数组的计算(广播机制：将数组中每一个元素都执行此运算)
print(arr,arr.shape,type(arr),arr.dtype)
arr2=numpy.array([i for i in range(1,9)],dtype=numpy.float64)
arr2/=2
print(arr2,arr2.shape,type(arr2),arr2.dtype)
arr3=arr2-arr       #数组间的计算(前提：数组同型)(广播机制：数组中每一个对应次序的元素进行计算)
print(arr3,arr3.shape,type(arr),arr.dtype)
arr1=numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
arr2=numpy.array([[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12]])
arr=arr1-arr2
print(arr,arr.shape,type(arr),arr.dtype)
arr_row=numpy.array([2,4,6,8])
arr_column=numpy.array([3,6,9]).reshape(3,1)
arr=arr-arr_row     #矩阵与行向量运算
print(arr,arr.shape,type(arr),arr.dtype)
arr=arr-arr_column      #矩阵与列向量的运算
print(arr,arr.shape,type(arr),arr.dtype)
arr=arr*arr_row     #矩阵与行向量的运算
print(arr,arr.shape,type(arr),arr.dtype)
arr=numpy.arange(1,25)
arr=arr.reshape(2,3,4)
arr=arr-arr_row     #三维数组与行向量的运算
print(arr,arr.shape,type(arr),arr.dtype)
arr=arr-arr_column      #三维数组与列向量的运算
print(arr,arr.shape,type(arr),arr.dtype)
arrr=numpy.arange(1,13).reshape(3,4)
arr=numpy.arange(1,25).reshape(2,3,4)
arr=arr-arrr        #三维数组与二维数组的运算
print(arr,arr.shape,type(arr),arr.dtype)