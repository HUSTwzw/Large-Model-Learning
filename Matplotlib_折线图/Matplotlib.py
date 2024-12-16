# 1.简单绘制折线图


from matplotlib import pyplot
x=range(2,26,2)
y=[15,13,14,5,17,20,15,26,26,27,22,18]
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.plot(x,y)
pyplot.xticks(x)
pyplot.yticks(range(min(y),max(y)+1))
pyplot.savefig("./picture1.png")
pyplot.show()


# 2.在1的基础上修改折线图的x、y轴坐标间距


from matplotlib import pyplot
import random
x=range(1,25)
y=[random.randint(15,30) for i in range(24)]
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.plot(x,y)
pyplot.xticks(x)
pyplot.yticks(range(min(y),max(y)+1))
pyplot.savefig("./picture2.png")
pyplot.show()


# 3.在2的基础上给坐标轴和轴坐标添加中文，创建网格背景


from matplotlib import pyplot
import random
from matplotlib import font_manager
x=range(0,120)
y=[random.randint(10,20) for i in x]
my_font=font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc")       #根据系统字体进行中文字体设置(在Linux系统终端(或者vscode的终端)输入fc-list :lang=ch，查看系统支持的字体)
x_ticks=["{}点{}分".format((6+int(i/60)),i%60) for i in x]
y_ticks=["{}℃".format(i) for i in range(min(y),max(y)+1)]
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.plot(x,y)
pyplot.xticks(list(x)[::4],x_ticks[::4],rotation=90,fontproperties=my_font)    #第一个参数和第二个参数需要一一对应,即数量相同,第三个参数rotation=顺时针旋转的度数，第四个参数表示应用设置的中文字体
pyplot.yticks(range(min(y),max(y)+1),y_ticks,rotation=90,fontproperties=my_font)
pyplot.xlabel("时间(6-8点)",fontproperties=my_font)
pyplot.ylabel("温度(℃)",fontproperties=my_font)
pyplot.grid(alpha=0.6)      #根据设置的x、y轴坐标间隔绘制网格，参数对应网格的透明度
pyplot.savefig("./picture3.png")
pyplot.show()


# 4.创建多条折线，设置折线名称，设置折线类型


from matplotlib import pyplot
import random
from matplotlib import font_manager
year=[i for i in range(18,29)]
book_1=[random.randint(0,7) for i in range(11)]
book_2=[random.randint(0,7) for i in range(11)]
my_font=font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-DemiLight.ttc")
year_list=["{}岁".format(i) for i in year]
book_list=["{}本".format(i) for i in range(min(min(book_1),min(book_2)),max(max(book_1),max(book_2))+1)]
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.xticks(year,year_list,fontproperties=my_font)
pyplot.yticks(range(min(min(book_1),min(book_2)),max(max(book_1),max(book_2))+1),book_list,rotation=90,fontproperties=my_font)
pyplot.xlabel("年龄(岁)",fontproperties=my_font)
pyplot.ylabel("所读书籍数量(本)",fontproperties=my_font)
pyplot.title("书籍阅读折线图",fontproperties=my_font)
pyplot.plot(year,book_1,label="学生1",color="blue",linestyle="--")      #给折线图添加一个标签(名称)，用于与其他折线图进行区分，同时可以设置字体颜色与种类
pyplot.plot(year,book_2,label="学生2",color="green",linestyle="-.")
pyplot.legend(prop=my_font)     #添加图例并设置为对应的中文字体     注意：只有legend才用prop接受my_font，其余都用fontproperties
pyplot.grid(alpha=0.6)
pyplot.savefig("./picture4.png")
pyplot.show()