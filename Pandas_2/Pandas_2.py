import pandas
import numpy
import string


pd1=pandas.DataFrame(numpy.arange(1,13).reshape(3,4))       #创建一个带行标签、列标签的三行四列的二维数组
print(pd1)
print(pd1.index)        #输出行标签
print(pd1.columns)      #输出列标签


pd2=pandas.DataFrame(numpy.arange(1,13).reshape(3,4),
                     index=[string.ascii_uppercase[i] for i in range(3)],
                     columns=[string.ascii_lowercase[i] for i in range(4)])     #创建一个以大写字母为行标签、以小写字母为列标签的三行四列的二维数组
print(pd2)
print(pd2.index)
print(pd2.columns)


stu_dict={"name":["Lucy","Leo","Jack"],"ID":["13488","13490","13469"],"age":[17,18,19],"address":["Wuhan","Beijing","Henan"]}
pd3=pandas.DataFrame(stu_dict)      #在DataFrame传入字典后，字典的key作为二维数组的行标签，每一个value列表作为其中一列
print(pd3)
print(pd3.index)
print(pd3.columns)


stu_list=[{"name":"Lucy","ID":"13488","age":17,"address":"Wuhan"},{"name":"Leo","ID":"13490","age":18,"address":"Beijing"},
          {"name":"Jack","ID":"13469","age":19,"address":"Henan"}]
pd4=pandas.DataFrame(stu_list)      #pd4的用法与pd3类似
print(pd4)
print(pd4.shape)        #输出数组的形状
print(pd4.ndim)         #输出数组的维数
print(pd4.index)        
print(pd4.columns)
print(pd4.values)       #输出数组的值
print(pd4.dtypes)       #输出列数据的类型


stu_list2=[{"name":"Lucy","ID":"13488","address":"Wuhan"},{"name":"Leo","ID":"13490","age":18,"address":"Beijing"},{"name":"Jack","address":"Henan"}]
pd5=pandas.DataFrame(stu_list2)
print(pd5)
print(pd5.dtypes)       #此时age类型为float，因为有NAN


pd6=pandas.read_csv("601939.csv")       #读取csv类型的数据，返回DataFrame类型的二维数组       
print(pd6)
print(pd6.dtypes)
print(pd6.columns)
print(pd6.index)
print(pd6.head(10))      #输出指定的前n行(默认为前5行)
print(pd6.tail())       #输出指定的前n行(默认为倒数5行)
print(pd6.info())       #输出相关信息概览：行数，列数，列索引，列非空值个数，列类型，内存占用情况
print(pd6.describe())       #输出综合统计结果：计数，均值，标准差，最小值，25%位数，50%位数，75%位数，最大值


pd7=pd6.sort_values(by="成交量",ascending=False)        #by表示排序对象，ascending表示升序
print(pd7)      #以成交量的逆序进行输出
pd7=pd6[3:8]        #选取第三至七行(行号从零开始排序)
print(pd7)
pd7=pd6["成交量"]       #选取"成交量"这一列
print(pd7)
print(type(pd7))        #注意：当选取单列时pd2为Sereis类型，印证DataFrame是由Series的基本单位构成
pd7=pd6[["开盘价","收盘价","成交量"]]       #选取多列
print(pd7)
pd7=pd6[3:8]["成交量"]      #同时选取行与列
print(pd7)


pd8=pd6.loc[[2,4,8],["成交量","成交金额"]]      #通过标签选取特定行列 
print(pd8)
pd8=pd6.loc[5:10,["成交量","成交金额","振幅"]]
print(pd8)
pd8=pd6.loc[5:10,"成交量":"换手率"]     #在loc函数中,行标签与列标签的:是闭区间(包含右区间)
print(pd8)


pd9=pd6.iloc[0:5,0:5]       #通过坐标选取特定行与列
print(pd9)
pd9=pd6.iloc[0:5,[4,3,2,1,0]]       #通过坐标选取特定行与列
print(pd9)
pd9=pd6.iloc[0:5,0:5]       #通过坐标选取特定行与列
pd9.iloc[2:4,[0,2,4]]=numpy.nan     #通过坐标选取特定行与列,并将其修改为NAN
print(pd9)

pd10=pd6[pd6["成交量"]>700000]      #将成交量>700000的所有行构成一个新数组
print(pd10)
print(pd10.describe())
pd10=pd6[(pd6["成交量"]>700000)&(pd6["成交量"]<800000)]         #将成交量>700000且<800000的所有行构成一个新数组
print(pd10)
print(pd10.describe())
pd10=pd6[(pd6["成交量"]<200000)|(pd6["成交量"]>2000000)]        ##将成交量<200000或>2000000的所有行构成一个新数组
print(pd10)
pd10=pd6[pd6["日期"].str.len()<5]       #将所有日期字符串长度<6的行组成一个新数组
print(pd10)         #输出新数组(由于日期字符串长度均为10,因此数组为空数组)

pd11=pd6.loc[0:5,"日期":]
pd11.iloc[1:3,2:5]=numpy.nan
print(pd11)
print(pandas.isnull(pd11))      #返回布尔类型的数组，将NAN视为True，其余视为False
print(pandas.notnull(pd11))     #返回布尔类型的数组，将NAN视为False，其余视为True
pd11.dropna(axis=0,how="any",inplace=True)      #axis=0表示对行进行删除，axis=1表示对列进行删除，how="any"表示只要含有一个NAN就进行删除，how="all"表示行/列全为NAN才能删除
                                                #inplace=True表示直接对pd1进行删除操作，inplace=False表示不对pd1进行删除操作，只将修改结果返回
print(pd11)
pd11=pd6.loc[0:5,"日期":]
pd11.iloc[1:3,2:5]=numpy.nan
pd11=pd11.dropna(axis=1,how="any",inplace=False)        
print(pd11)

pd11=pd6.loc[0:5,"开盘价":]
pd11.iloc[1:3,2:5]=numpy.nan
print(pd11.fillna(pd11.mean()))     #替换所有NAN:将NAN修改为每一列的均值
print(pd11)
pd11["收盘价"]=pd11["收盘价"].fillna(pd11["收盘价"].mean())     #替换指定NAN:将"收盘价"一列的NAN修改为此列均值
print(pd11)