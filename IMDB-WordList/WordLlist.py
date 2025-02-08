# 根据IMDB中2.5w个电影评论构建词表


import pandas as pd
import chardet
import json


def LoadIBMData(path):
    with open(path, "rb") as f:
        raw_data = f.read()     #二进制形式读取数据
        detected_encoding = chardet.detect(raw_data)["encoding"]        #输入二进制形式数据并得到编码格式
    print(detected_encoding)        #编码格式为ISO-8859-1
    data=pd.read_csv("imdb_tr.csv",encoding=detected_encoding)      #以特定编码形式读取数据
    return data


def ProcessData(data):      #加工数据的函数
    data=data["text"]
    data=data.str.lower()       #将评论小写化
    data=data.str.replace(r"[^a-z0-9']"," ",regex=True)        #通过正则化除去字母、数字、单引号之外的符号,并替换为空格
    data=data.str.replace(r"'[a-z0-9]*"," ",regex=True)        #通过正则化将所有以单引号开头的字符串替换为空格
    data=data.str.replace(r"[' ']+"," ",regex=True)            #通过正则化将空格数量确定为1个
    data=data.str.split(" ")
    return data
    
    
def CreateWordList(data):       #创造词表的函数
    word_list={}
    line=data.shape[0]
    for i in range(line):
        for word in data[i]:
            if word not in word_list:
                word_list[word]=1
            elif word in word_list:
                word_list[word]+=1
    del word_list['']
    return word_list


def SortedWordList(word_list):      #将词表按照出现次数逆序排列的函数
    sorted_word_list={}
    sorted_word_count=sorted(word_list.items(),key=lambda x:x[1],reverse=True)
    for word,count in sorted_word_count:
        sorted_word_list[word]=count
    return sorted_word_list
      

def SelectedWordList(sorted_word_list,k):       #从词表筛选一定出现次数的单词构建新词表
    selected_word_list={}
    for key,value in sorted_word_list.items():
        if value>=k:
            selected_word_list[key]=value
        else:
            return selected_word_list


def FinalWordList(selected_word_list):      #最终的词表
    final_word_list={}
    num=len(selected_word_list.keys())
    for i in range(num):
        final_word_list[int(i)]=list(selected_word_list.keys())[i]
    return final_word_list


def TextToSequence(data,final_word_list):       #将原电影评论以词表序号替代
    vocab={word:int(idx) for idx,word in final_word_list.items()}
    index=data.shape[0]
    sequences=[]
    for i in range(index):
        sequence=[]
        for word in list(data[i]):
            if word in final_word_list.values():
                sequence.append(vocab[word])
            else:
                sequence.append(-1)     #将不在词表的单词统一设置为-1
        sequences.append(sequence)
    return sequences
            

path="imdb_tr.csv"
data=LoadIBMData(path)
data=ProcessData(data)
word_list=CreateWordList(data)
sorted_word_list=SortedWordList(word_list)
selected_word_list=SelectedWordList(sorted_word_list,100)       #将出现次数>=100的单词构成词表
final_word_list=FinalWordList(selected_word_list)       #为词表中每个单词编号
sequences=TextToSequence(data,final_word_list)      #将原评论单词以编号进行替换
with open("SortedWordList.json","w",encoding="utf-8") as json_file:
    json.dump(sorted_word_list,json_file,indent=4)
with open("SelectedWordList.json","w",encoding="utf-8") as json_file:
    json.dump(selected_word_list,json_file,indent=4)
with open("FinalWordList.json","w",encoding="utf-8") as json_file:
    json.dump(final_word_list,json_file,indent=4)
with open("ProcessedIBM.json","w",encoding="utf-8") as json_file:
    json.dump(sequences,json_file,indent=4) 