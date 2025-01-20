# 时间序列
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager


#创建timeIndex(日期索引):date_range
date1=pd.date_range(start="20250119",end="20250216",freq="D")       #开始时间为2025-01-19,结束时间为2025-02-16,间隔为一天
print(date1)        
print(type(date1))      #类型为DatetimeIndex
print(date1.dtype)      #其中每一个元素类型为datetime64[ns]
date2=pd.date_range(start="20250119",end="20250216",freq="3D")      #间隔为三天
print(date2)
date3=pd.date_range(start="20250119",periods=10,freq="D")       #periods的值即生成的日期个数
print(date3)
date4=pd.date_range(start="20250119",periods=10,freq="3D")      #以2025-01-19为起始日期,生成十个日期,日期间隔为三天
print(date4)
"""
补充:常见freq:
D:每天
B:每工作日
H:每小时
T:每分钟
S:每秒
M:每月最后一个日历日
BM:每月最后一个工作日
MS:每月第一个日历日
BMS:每月第一个工作日
"""


#将数据转换为datatime对象:to_datetime
pd1=pd.read_csv("601939.csv")
time_series=pd1["日期"]
time_list=pd1["日期"].values
timedata=pd.to_datetime(time_series,format="%Y-%m-%d")        #to_datetime方法可以将Series或者列表类型转换为datetime对象,其中format表示时间格式
print(type(timedata))       #如果提供的数据是Series,则timedata是Series类型
print(timedata)
timedata2=pd.to_datetime(time_list,format="%Y-%m-%d")
print(type(timedata2))      #如果提供的数据是列表,则timedata是DatetimeIndex类型
print(timedata2)


#重采样:resample
pd1=pd1[["成交金额"]]       #pd1改为只有一列的Dataframe
data=pd1.set_index(timedata)        #将index设置为DatetimeIndex
print(data.resample("M").sum())        #注意:resample需要作用在具有DatetimeIndex(每个元素是datetime)的Series或者DataFrame上
#注意:重采样后可以使用sum\mean\std\var\count\max\min等函数进行数据分析


#综合练习：以20天为单位,绘制成交量的折线图
#处理数据:
pd2=pd.read_csv("601939.csv")
Time_series=pd2["日期"]
Timedata=pd.to_datetime(Time_series,format="%Y-%m-%d")
pd2=pd2[["成交量"]]
pd2=pd2.set_index(Timedata)
Data=pd2.resample("20D").sum()
#绘图:
myfont=font_manager.FontProperties("Microsoft YaHei")
plt.figure(figsize=(12,9),dpi=80)
x_ticks=[i.strftime("%Y-%m-%d") for i in Data.index]        #将datetime转换为特定格式的字符串
plt.xticks(range(len(x_ticks)),x_ticks,rotation=90,fontproperties=myfont)
plt.plot(range(len(x_ticks)),Data["成交量"].values)
plt.ylim(0,2.5e7)       #确定y轴边界(最小值与最大值)
plt.xlabel("日期",fontproperties=myfont,loc="right",labelpad=-20)       #loc负责确定左右位置,labelpad负责上下移动
plt.ylabel("股票交易量(股)",fontproperties=myfont)
plt.title("股票交易量折线图",fontproperties=myfont)
plt.savefig("picture1.png")
plt.show()      #y轴数值太大,自动出现科学计数法