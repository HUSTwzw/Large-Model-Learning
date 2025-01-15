# 数据离散化
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager


pd1=pd.read_csv("beijing.csv",nrows=5200)
print(pd1.info())
list1=pd1["中类"].tolist()      #将dataframe特定一行转换为列表
print(list1)
set_list1=set(list1)        #转化为set类型(列表中每一个不同的元素组合在一起,类似于取集合)
print(set_list1)
print(len(set_list1))       #统计中类的种类数量(方法一)
unique_pd1=pd1["中类"].unique()         #获取Series(或DataFrame中的某一列)中的唯一值(去重后的值),它返回一个包含唯一值的NumPy数组
print(unique_pd1,type(unique_pd1))
print(len(unique_pd1))      #统计中类的种类数量(方法二)
print(pd1["经度"].mean())       #平均经度
print(pd1["纬度"].mean())       #平均纬度
print(pd1["经度"].argmax())     #经度最大值的行标
print(pd1["经度"].argmin())     #经度最小值的行标
print(pd1["经度"].max())        #经度最大值
print(pd1["经度"].min())        #经度最小值
print(pd1["经度"].median())     #经度中位数


row=pd1.shape[0]        #行数
column=len(pd1["中类"].unique())
print(column)
pd2=pd.DataFrame(np.zeros((row,column)),index=range(row),columns=pd1["中类"].unique())        #生成零矩阵
for i in range(row):
    pd2.loc[i,pd1["中类"][i]]=1     #将对应的中类种类赋值为1
result=pd2.sum(axis=0)      #对每一个中类种类进行求和(每列分别求和),以此实现对每一个中类的数量统计,返回series类型
result=result.sort_values()     #将result(series类型)按values正序进行排序
print(result)

my_font=font_manager.FontProperties("Microsoft YaHei")      #绘图(条形图)
plt.figure(figsize=(19,9),dpi=80)
x_ticks=[i for i in result.index]
plt.xticks(range(result.shape[0]),x_ticks,rotation=90,fontproperties=my_font)
plt.bar(result.index,result.values)
plt.ylabel("数量(个)",fontproperties=my_font)
plt.savefig("picture1.png")
plt.show()