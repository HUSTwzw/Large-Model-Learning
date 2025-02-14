# 通过Dataset和Dataloader包装数据


import torch
from torch.utils.data import Dataset,DataLoader
import chardet
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
import numpy as np
import json


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


class CSVMovieDataset(Dataset):
    def __init__(self,csvfilename,wordlist=None,maxlength=500):
        
        with open(csvfilename,"rb") as f:
            raw_data=f.read()       #二进制形式读取数据      
            detected_encoding=chardet.detect(raw_data)["encoding"]      #输入二进制形式数据并得到编码格式
        
        self.data=pd.read_csv(csvfilename,encoding=detected_encoding)       #以特定编码形式读取数据
        
        self.wordlist=wordlist      #词表
        
        self.maxlength=maxlength        #数据最大长度
        
        if self.wordlist==None:
            self.wordlist=self.buildwordlist()      
            
        self.texts,self.labels=self.processtext()
            
                
    def buildwordlist(self):        #
        wordlist={"UNK":0}      #0号词条表示unknown的单词(即不重要或不常在评论中出现的单词)
        
        stop_words=set(stopwords.words("english"))   
        stemmer=SnowballStemmer("english")
        
        for text in self.data["text"]:
            text=re.sub("[^a-zA-Z]"," ",text)      #将非字母转换为空格
            text=text.lower().strip()       #小写化并去除首尾的空格
            words=word_tokenize(text)       #自动分词
            words=[stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]       #将非停用词词干化
            for word in words:
                if word not in wordlist:
                    wordlist[word]=len(wordlist)
                    
        return wordlist
    

    def processtext(self):      #依据词表将评论以数字矩阵形式替换
        labels=torch.tensor(self.data["polarity"].values,dtype=torch.long)
        
        stop_words=set(stopwords.words("english"))
        stemmer=SnowballStemmer("english")
        
        texts=[]
        
        for text in self.data["text"]:
            text=re.sub("[^a-zA-Z]"," ",text)
            text=text.lower().strip()
            words=word_tokenize(text)
            words=[stemmer.stem(word) for word in words if word not in stop_words and word.isalpha()]
            indices=[self.wordlist.get(word,0) for word in words]
            
            if len(indices)>self.maxlength:     #截断数据
                indices=indices[:self.maxlength]
            elif len(indices)<self.maxlength:       #补充无效数据
                indices=indices+[0]*(self.maxlength-len(indices))
            
            texts.append(torch.tensor(indices,dtype=torch.long))
            
        return texts,labels
    
    
    def __len__(self):      
        return len(self.data)       #返回行数
    
    def __getitem__(self,idx):
        return self.texts[idx],self.labels[idx]     #返回特定行数据
    
    
def SaveToJson(dataloader):     #保存为json格式(数据是乱序版)
    data_list=[]
    
    for batch_idx,(texts,labels) in enumerate(dataloader):
        batch_texts=np.array(texts).tolist()
        batch_labels=np.array(labels).tolist()
        for text,label in zip(batch_texts,batch_labels):
            data={"text":text,"label":label}
            data_list.append(data)
            
    with open("data.json","w",encoding="utf-8") as json_file:
        json.dump(data_list,json_file,indent=4)
            
    
def main():
    csvfilename="imdb_tr.csv"
    
    dataset=CSVMovieDataset(csvfilename)
    
    dataloader=DataLoader(dataset,batch_size=128,shuffle=True,num_workers=4)        #128个数据为一次,数据乱序重组
    
    for batch_idx,(texts,labels) in enumerate(dataloader):      #迭代
        print(f"Batch{batch_idx+1}")
        print(f"TextsShape:{texts.shape}")
        print(f"LabelsShape:{labels.shape}")

    SaveToJson(dataloader)

if __name__=="__main__":
    main()        