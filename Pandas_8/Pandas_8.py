# 综合练习:
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import font_manager
import numpy as np


#统计北京市各区"科教文化"大类的数量分布(以升序形式呈现)
#处理数据:
pd1=pd.read_csv("beijing.csv")
data1=pd1.set_index(["大类"])       #将大类作为index
data1=data1.loc["科教文化"]     #从大类中筛选出"科教文化"
grouped1=data1.groupby(by="区域").count()       #统计各区"科教文化"数量,此时为Series类型
grouped1=grouped1["名称"]       #选取其中一列,实现从DataFrame到Series类型的转换
grouped1=grouped1.sort_values(ascending=False)      #以降序形式进行排序
#绘图:
myfont=font_manager.FontProperties("Microsoft YaHei")
plt.figure(figsize=(12,8),dpi=80)
x_ticks=grouped1.index
plt.bar(range(len(x_ticks)),grouped1.values)
plt.xticks(range(len(x_ticks)),x_ticks,fontproperties=myfont)       #调整x轴标签
plt.yticks(range(0,14000,2000))     #调整y轴间隔
plt.xlabel("地区(区)",fontproperties=myfont)
plt.ylabel("数量(家)",fontproperties=myfont)
plt.title("各区科教文化数量分布",fontproperties=myfont)
plt.grid(axis="y",alpha=0.8)        #绘制y方向网格
plt.savefig("picture1.png")
plt.show()


#统计海淀区"休闲娱乐"大类下各个中类的数量分布
#处理数据:
pd2=pd.read_csv("beijing.csv")
data2=pd2[pd2["区域"]=="海淀区"]        #选取海淀区
data2=data2[data2["大类"]=="休闲娱乐"]      #选取"休闲娱乐"大类
grouped2=data2.groupby(by="中类").count()["名称"]       #统计各种类数量并将其转化为Series
grouped2=grouped2.sort_values(ascending=False)      #以降序顺序进行排序
#绘图:
myfont=font_manager.FontProperties("Microsoft YaHei")
plt.figure(figsize=(12,8),dpi=80)
x_ticks=grouped2.index
plt.bar(range(len(x_ticks)),grouped2.values)
plt.xticks(range(len(x_ticks)),x_ticks,fontproperties=myfont)
plt.yticks(range(0,450,50))
plt.xlabel("休闲娱乐部门",fontproperties=myfont)
plt.ylabel("数量(家)",fontproperties=myfont)
plt.title("海淀区休闲娱乐产业数量分布",fontproperties=myfont)
plt.grid(axis="y",alpha=0.8)
plt.savefig("picture2.png")
plt.show()


#统计交通设施地理位置
pd3=pd.read_csv("beijing.csv")
data3=pd3[pd3["大类"]=="交通设施"]      
data3=data3[["纬度","经度","区域"]]
grouped3=data3.groupby(by="区域").mean()        #计算各区交通设施的平均经纬度,得出每一个区的交通设施中心
latitude=grouped3["纬度"]       #获取各区平均纬度信息
longitude=grouped3["经度"]      #获取各区平均经度信息
myfont=font_manager.FontProperties("Microsoft YaHei")
plt.figure(figsize=(12,8),dpi=80)
plt.scatter(longitude,latitude)
plt.xticks(np.linspace(115.8,117.2,8))      #调控x轴(经度)取值范围
plt.yticks(np.linspace(39.5,40.5,11))       #调控y轴(纬度)取值范围
plt.xticks(np.linspace(115.8,117.2,8),["{}°E".format(i) for i in np.linspace(115.8,117.2,8)],fontproperties=myfont)
plt.yticks(np.linspace(39.5,40.5,11),["{}°N".format(i) for i in np.linspace(39.5,40.5,11)],fontproperties=myfont)
plt.grid(alpha=0.8)
plt.xlabel("经度",fontproperties=myfont)
plt.ylabel("维度",fontproperties=myfont)
plt.title("北京市交通设施中心分布图",fontproperties=myfont)
plt.savefig("picture3.png")
plt.show()