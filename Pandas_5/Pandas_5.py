# 数据合并
import pandas as pd
import numpy as np
import string


# 按行合并：join
pd1=pd.DataFrame(np.arange(1,13).reshape(3,4),
                 index=[string.ascii_lowercase[i] for i in range(3)],
                 columns=[string.ascii_uppercase[i] for i in range(4)])
pd2=pd.DataFrame(np.arange(13,21).reshape(2,4),
                 index=[string.ascii_lowercase[i] for i in range(2)],
                 columns=[string.ascii_uppercase[i] for i in range(4)])
print(pd1)
print(pd2)
pd3=pd1.join(pd2,lsuffix="_pd1",rsuffix="_pd2")     #合并pd1与pd2时列号不能重叠,因此需要分别加上后缀
print(pd3)      #行数以pd1为标准,多余部分填充NAN
pd4=pd2.join(pd1,lsuffix="_pd2",rsuffix="_pd1")
print(pd4)      #行数以pd2为标准(pd1部分行被切割)

# 按列合并:merge
"""
核心参数:
how:合并方式,默认为"inner"可选值包括:
    "inner":求交集
    "outer":求并集
    "left":左连接(保留左侧DataFrame的所有行)
    "right":右连接(保留右侧DataFrame的所有行)
on:用于合并的列名(两个DataFrame中列名相同)
left_on:左侧DataFrame中用于合并的列名
right_on:右侧DataFrame中用于合并的列名
suffixes:如果列名冲突,为左右DataFrame的列名添加后缀,默认为('_x','_y')(自动添加)

工作原理:
选择键:确定用于匹配的列或索引
匹配行:根据键的值,在两个DataFrame中查找匹配的行
合并方式:根据指定的合并方式(如内连接、左连接等),决定如何保留或丢弃不匹配的行
生成结果:将匹配的行合并为一个新的DataFrame
"""
pd5=pd.DataFrame(np.eye(3),
                 index=[string.ascii_lowercase[i] for i in range(3)],
                 columns=[string.ascii_uppercase[i] for i in range(3)])
pd6=pd.DataFrame(np.ones((3,5)),
                 index=[string.ascii_lowercase[i] for i in range(3)],
                 columns=[string.ascii_uppercase[i] for i in range(5)])
print(pd5)
print(pd6)
print(pd5.merge(pd6,on="A",how="inner"))        #依据pd5和pd6的A列:pd5的a行匹配pd6的a、b、c行(交集)
print(pd5.merge(pd6,on="A",how="outer"))        #依据pd5和pd6的A列:pd5的a行匹配pd6的a、b、c行,同时pd5的b、c行不匹配pd6,以NAN填充(并集)       
print(pd5.merge(pd6,on="A",how="left"))         #依据pd5和pd6的A列:保留pd5所有行,寻找与pd5匹配的pd6的行
print(pd5.merge(pd6,on="A",how="right"))        #依据pd5和pd6的A列:保留pd6所有行,寻找与pd6匹配的pd5的行
print(pd5.merge(pd6,left_on="B",right_on="A",how="inner"))      #依据pd5的B列、pd6的A列:pd5的b行匹配pd6的a、b、c行(交集)
print(pd5.merge(pd6,left_on="B",right_on="A",how="outer"))      #依据pd5的B列、pd6的A列:pd5的b行匹配pd6的a、b、c行,同时pd5的b、c行不匹配pd6,以NAN填充(并集)
print(pd5.merge(pd6,left_on="B",right_on="A",how="left"))       #依据pd5的B列、pd6的A列:保留pd5所有行,寻找与pd5匹配的pd6的行
print(pd5.merge(pd6,left_on="B",right_on="A",how="right"))      #依据pd5的B列、pd6的A列:保留pd6所有行,寻找与pd6匹配的pd5的行