# 数据的分组与聚合
import pandas as pd


pd1=pd.read_csv("beijing.csv",nrows=52000)
print(pd1)
pd2=pd1.groupby(by="大类")      #以"大类"为分类依据进行分组
print(pd2)      
print(type(pd2))        #pd2为一种特殊类型(DataFrameGroupBy对象),包含一个个元组,元组的第一个值为"大类"的具体种类,第二个值是dataframe
for i,j in pd2:
    print(i)
    print(j)
print(pd2.count())      #将每一个"大类"进行数量统计
print(pd2["区域"].count())      #统计区域中每一个大类的数量
print(pd2["区域"].count()["休闲娱乐"])      #统计区域中"休闲娱乐"的数量
print(pd2["区域"].count()["科教文化"])      #统计区域中"科教文化"的数量
# 注意:采用groupby和count方法十分便捷,原始方法详见Github:Large-Model_Learning仓库:Pandas_4(创建零矩阵、对应位置填充1、对列使用sum方法)
# 补充:关于选取特定大类的所有行,除了使用groupby,也可以直接通过bool索引选取,例如pd1[pd1["大类"]=="休闲娱乐"]


pd3=pd1[pd1["大类"]=="旅游景点"]        #选取"大类"为"旅游景点"的行
print(pd3.groupby(by="区域").count())       
print(pd3.groupby(by="区域").count()["城市"])       #统计旅游景点在每一个区的分布数量


"""
DataFrameGroupBy常见方法总结:
count:统计非NAN的数量
sum:统计非NAN的和
mean:统计非NAN的平均值
median:统计非NAN的中位数
std:统计非NAN的标准差
var:统计非NAN的方差
min:统计最小值
max:统计最大值
"""


for i,j in pd1.groupby(by=["大类","中类"]):     #将"大类"与"中类"的不同组合作为分组依据
    print(i)
    print(j)
grouped1=pd1.groupby(by=["大类","中类"]).count()        #统计每一种组合的数量
grouped2=pd1.groupby(by=["大类","中类"]).count()["区域"]        #选取每一种组合的特定一列
grouped3=pd1.groupby(by=["大类","中类"]).count()[["区域"]]      #选取每一种组合的特定一列
print(grouped1)
print(type(grouped1))
print(grouped2)     
print(type(grouped2))       #单一方括号表明选取一列,因此类型为Series
print(grouped3)
print(type(grouped3))       #嵌套方括号在语法上表示选取多列,因此类型为DataFrame