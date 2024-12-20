# 1.绘制直方图(纵坐标为个数)


from matplotlib import pyplot
from matplotlib import font_manager
num=[80,69,88,97,105,78,89,100,113,107,108,73,103,101,74,118,120,99,102,72,128,129,87,94,96,75,78,99,119,121]
step=5
amount=(max(num)-min(num))/step     #两个整数相除得到浮点数，而pyplot.hist的第二个参数必须是int型   
amount=int(amount)
my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
x_ticks=["{} mins".format(i) for i in range(min(num),max(num)+step,step)]
y_ticks=["{}人".format(i) for i in range(0,8)]
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.hist(num,amount)     #绘制直方图，需要注意：如果max(num)-min(num)不能整除step，则绘制的图形会发生错位    #原理：根据amount和max(num)与min(num)确定绘图间距，并统计每一个区间(左闭右开)的个数
pyplot.xticks(range(min(num),max(num)+step,step),x_ticks)   #参数ticks要与绘图时的横坐标大体对应，否则会产生图形严重偏离(导致直方图不能居中)
pyplot.yticks(range(0,8),y_ticks,rotation=90,fontproperties=my_font)
pyplot.xlabel("阅读时间",fontproperties=my_font)
pyplot.ylabel("人数",fontproperties=my_font)
pyplot.title("阅读时间统计图",fontproperties=my_font)
pyplot.grid(alpha=0.8)
pyplot.savefig("./picture1.png")
pyplot.show()


# 2.绘制直方图(纵坐标为频率)


from matplotlib import pyplot
from matplotlib import font_manager 
num=[80,69,88,97,105,78,89,100,113,107,108,73,103,101,74,118,120,99,102,72,128,129,87,94,96,75,78,99,119,121]
percent=[round(i*0.025,4) for i in range(0,11)]      #percent是针对频率的刻度间隔
percent_density=[round(i/5,4) for i in percent]     #percent_density是针对概率密度的刻度间隔
step=5
amount=(max(num)-min(num))/step
amount=int(amount)
my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
x_ticks=["{}mins".format(i) for i in range(min(num),max(num)+step,step)]
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.hist(num,amount,density=True)    #获得的是概率密度(概率密度=频率/组间距，即概率密度=频率/step)
pyplot.xticks(range(min(num),max(num)+step,step),x_ticks)
pyplot.yticks(percent_density,percent)      #由于直方图是根据概率密度所绘制，因此思路是保留概率密度的刻度间隔，并将其替换为频率对应的标签
pyplot.xlabel("阅读时间",fontproperties=my_font)
pyplot.ylabel("频率",fontproperties=my_font)
pyplot.title("阅读时间分布图",fontproperties=my_font)
pyplot.grid(alpha=0.8)
pyplot.savefig("./picture2.png")
pyplot.show()