import numpy
arr1=numpy.array([i for i in range(1,6)])       #使用numpy创建数组的常见方法
arr2=numpy.array(range(1,6))
arr3=numpy.arange(1,6)
arr4=numpy.array(range(1,8,2))
arr5=numpy.arange(1,8,2)
print(arr1,type(arr1))
print(arr2,type(arr2))
print(arr3,type(arr3))
print(arr4,type(arr4))
print(f"{arr5}  {type(arr5)}  {arr5.dtype}")       #使用type()可以查看数组的类型(numpy.ndarray),使用.dtype可以查看数组内元素的类型(int64,表示64位整型)
#int64是numpy特有的类型,包括:int8/16/32/64(有符号整型),uint8/16/32/64(无符号整型),float16/32/64/128(浮点型),complex64/128/256(复数),bool(布尔型)
arr6=numpy.array([1,0,0,1,0,1,1,1,0,0,1],dtype="bool")        #将数组中的元素指定为布尔型(格式1)
print(f"{arr6}  {type(arr6)}  {arr6.dtype}")
arr6=numpy.array([0,1,1,1,0,0,1,0,0],dtype=numpy.bool_)     #将数组中的元素指定为布尔型(格式2)
print(f"{arr6}  {type(arr6)}  {arr6.dtype}")
arr7=numpy.arange(1,8,2,dtype="float64")     #将数组中的元素指定为float64类型(格式1)
print(f"{arr7}  {type(arr7)}  {arr7.dtype}")
arr7=numpy.arange(10,1,-2,dtype=numpy.float64)      #将数组中的元素指定为float64类型(格式2)
print(f"{arr7}  {type(arr7)}  {arr7.dtype}")
arr8=arr6.astype("int64")       #将已创建数组中的元素类型修改为int型(格式1)
print(f"{arr8}  {type(arr8)}  {arr8.dtype}")
arr8=arr6.astype(numpy.int32)       #将已创建数组中的元素类型修改为int型(格式2)
print(f"{arr8}  {type(arr8)}  {arr8.dtype}")
arr9=numpy.array([round(numpy.random.uniform(1.0,10.0),5) for i in range(8)],dtype=numpy.float64)       #numpy.random.uniform(a,b)随机生成处于[a,b)的浮点数
print(f"{arr9}  {type(arr9)}  {arr9.dtype}")