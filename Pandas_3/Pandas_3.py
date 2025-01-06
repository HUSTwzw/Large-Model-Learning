# 综合应用


import pandas
import numpy
import matplotlib
from matplotlib import pyplot
from matplotlib import font_manager


pd=pandas.read_csv("601939.csv")
pd1=pd["最高价"]        #此时为Series类型，若使用pd1=pd1.values，则变为ndarray类型
pd1=[round(i,3) for i in pd1]       #将数据转换为列表

step=0.02       #设置间距
amount=round((max(pd1)-min(pd1))/step,2)        #浮点数计算往往产生偏差，导致绘制的图形可能错位，因此需要使用四舍五入的函数
amount=int(amount)      #设置根据间距设置数量(注意：必须为int)
my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")       #设置字体
price_arr=numpy.arange(round(min(pd1),3),round(max(pd1)+step,3)-1e-10,step)     #由于数据为浮点数，不能使用range，只能通过numpy进行平均分割
                                                                                #浮点数由于精度问题可能超边界，因此可以通过减一个很小的数剔除超出的部分
pyplot.figure(figsize=(16,10),dpi=80)        #设置图片大小与清晰度
pyplot.hist(pd1,amount)     #绘制图形
x_ticks=["{}元/股".format(round(i,3)) for i in price_arr]       
y_ticks=["{}次".format(i) for i in range(16)]       
pyplot.xticks(price_arr,x_ticks,rotation=90,fontproperties=my_font)     #设置x轴
pyplot.yticks(range(0,16),y_ticks,fontproperties=my_font)       #设置y轴
pyplot.grid(alpha=0.8)      #绘制格子并设置清晰度
pyplot.xlabel("股票单价",fontproperties=my_font)        #设置x轴标签
pyplot.ylabel("出现频数",fontproperties=my_font)        #设置y轴标签
pyplot.title("股票单价-频数图",fontproperties=my_font)      #设置图形标题
pyplot.savefig("./picture1.png")        #保存图片到本地
pyplot.show()       #展示图片