# 1.创建竖直的条形图


from matplotlib import pyplot
from matplotlib import font_manager
name=["Lucy","Leo","小张","Mack","小王","小赵","Jerry","Jim"]
money=[320,2000,600,550,380,880,540,1300]
my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.xticks(range(len(name)),name,fontproperties=my_font)
pyplot.yticks(rotation=90)
pyplot.xlabel("姓名",fontproperties=my_font)
pyplot.ylabel("资产(W)",fontproperties=my_font)
pyplot.title("资产图",fontproperties=my_font)
pyplot.bar(name,money,width=0.5,color="blue")
pyplot.savefig("./picture1.png")
pyplot.show()


# 2.创建横向的条形图


from matplotlib import pyplot
from matplotlib import font_manager
name=["Lucy","Leo","小张","Mack","小王","小赵","Jerry","Jim"]
money=[320,2000,600,550,380,880,540,1300]
my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.yticks(range(len(name)),name,fontproperties=my_font)
pyplot.xlabel("资产(W)",fontproperties=my_font)
pyplot.ylabel("姓名",fontproperties=my_font)
pyplot.title("资产图",fontproperties=my_font)
pyplot.barh(name,money,height=0.5,color="green")        #使用bar函数时使用width,使用barh时使用height
pyplot.savefig("./picture2.png")
pyplot.show()


# 3.创建多个条形图


from matplotlib import pyplot
from matplotlib import font_manager
date=["12月3日","12月4日","12月5日"]
film={"电影1":[0.5,0.8,0.9],"电影2":[2.2,5.8,6.6],"电影3":[1.4,1.9,2.1],"电影4":[0.9,1.5,2.9]}
bar_width=0.2
film1=[i for i in range(len(date))]     #从film1到film4借助数字坐标确定多个柱状图的间隔
film2=[i+bar_width for i in range(len(date))]
film3=[i+bar_width*2 for i in range(len(date))]
film4=[i+bar_width*3 for i in range(len(date))]
middle=[]       #确定居中位置的坐标
for i in range(len(film2)):
    middle.append((film2[i]+film3[i])/2)
max_list=[max(i) for i in [film["电影1"],film["电影2"],film["电影3"],film["电影4"]]]
max_money=max(max_list)
y_ticks=["{}亿元".format(i) for i in range(int(max_money)+2)]
my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.bar(film1,film["电影1"],label="电影1",width=bar_width)       #依次绘制不同电影在12与3日到12月5日的票房
pyplot.bar(film2,film["电影2"],label="电影2",width=bar_width)
pyplot.bar(film3,film["电影3"],label="电影3",width=bar_width)
pyplot.bar(film4,film["电影4"],label="电影4",width=bar_width)
pyplot.legend(prop=my_font)
pyplot.xticks(middle,date,fontproperties=my_font)       #根据数字坐标确定日期居中位置       xticks(ticks,label,**kwargs)    其中ticks为位置列表(通过数字大小确定刻度的位置),label是所确定刻度对应的标签
pyplot.yticks(range(len(y_ticks)),y_ticks,rotation=90,fontproperties=my_font)
pyplot.xlabel("日期",fontproperties=my_font)
pyplot.ylabel("票房",fontproperties=my_font)
pyplot.title("电影票房-日期柱状图",fontproperties=my_font)
pyplot.savefig("./picture3.png")
pyplot.show()