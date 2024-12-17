# 1.绘制散点图


from matplotlib import pyplot
from matplotlib import font_manager 


a=[2,5,4,7,14,15,19,14,8,6,9,3]
b=[30,28,20,24,18,9,15,12,19,23,24,27]
month=["{}月".format(i) for i in range(1,13)]
ab=[i for i in range(min(min(a),min(b)),max(max(a),max(b))+4)]
ab_list=["{}万".format(i) for i in range(min(min(a),min(b)),max(max(a),max(b))+4)]


my_font=font_manager.FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
pyplot.figure(figsize=(12,8),dpi=80)
pyplot.xticks(range(12),month,fontproperties=my_font)
pyplot.yticks(ab[::3],ab_list[::3],rotation=90,fontproperties=my_font)
pyplot.scatter(month,a,label="种群1",color="blue")
pyplot.scatter(month,b,label="种群2",color="green")
pyplot.legend(prop=my_font)
pyplot.xlabel("时间(月)",fontproperties=my_font)
pyplot.ylabel("种群数量(万)",fontproperties=my_font)
pyplot.title("种群与时间关系图")
pyplot.savefig("./picture1.png")
pyplot.show()