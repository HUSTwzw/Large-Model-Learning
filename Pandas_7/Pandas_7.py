# 数据索引
import pandas as pd
import string
import numpy as np


pd1=pd.read_csv("beijing.csv")


# 获取索引
grouped1=pd1.groupby(by="大类").count()
print(grouped1.index)       #获取行索引组成的列表
print(grouped1.columns)     #获取列索引组成的列表
grouped2=pd1.groupby(by=["大类","中类"]).count()
print(grouped2.index)       #MultiIndex:复合索引
print(grouped2.columns)


# 指定索引
grouped1.index=[j for j in [string.ascii_uppercase[i] for i in range(len(grouped1.index))]]
print(grouped1)     #将行索引修改为大写字母
grouped1.columns=[j for j in [string.ascii_lowercase[i] for i in range(len(grouped1.columns))]]
print(grouped1)     #将列索引修改为小写字母


# 裁剪索引(reindex)
print(grouped1.reindex(["A","B","F","Q"]))      #原本存在"A"、"B"、"F"行,直接截取,不存在"Q"行,需要创建并统一赋值为NAN


# 某一列作为索引(set_index)
print(grouped1.set_index("b"))      #将"b"列作为index,同时删除"b"列
print(grouped1.set_index("b").index)        #输出新索引
print(grouped1.set_index("b",drop=False))       #将"b"列作为index,同时保留"b"列
# 注意:使用set_index产生的index可以重复
# 例子:
pd2=pd.DataFrame(np.eye(3),index=["a","b","c"],columns=["A","B","C"])
print(pd2.set_index("B").index)     #将"B"作为index,0.0重复出现
print(pd2.set_index("B").index.unique())        #返回不重复的index


# 以多列构成索引(set_index)
print(pd1.set_index(["大类","中类"]).index)
# 补充:对于两列index,若想要交换index,可以使用swaplevel()方法
# 例子:
print(pd1.set_index(["中类","大类"]))
print(pd1.set_index(["中类","大类"]).swaplevel())


# 利用索引获取数据:
# 方式一:
pd3=pd1.set_index(["大类","中类"])
pd4=pd3["区域"]     #pd4为Series类型
print(pd4)
print(type(pd4))        
print(pd4["交通设施"])
print(type(pd4["交通设施"]))        #Series类型      
print(pd4["交通设施"]["公交站"])
print(type(pd4["交通设施"]["公交站"]))      #Series类型
# 注意:如果pd4不是Series类型不能采取此方法,因为DataFrame默认选取列,因此需要使用loc方法

# 方式二:loc
pd5=pd3[["经度","纬度"]]        #pd5为DataFrame类型
print(pd5)
print(type(pd5))
print(pd5.loc["旅游景点"])
print(type(pd5.loc["旅游景点"]))
print(pd5.loc["旅游景点"].loc["公园"])
print(type(pd5.loc["旅游景点"].loc["公园"]))