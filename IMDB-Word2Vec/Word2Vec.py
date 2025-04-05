from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import re
import pandas as pd
import chardet


class my_word2vec():
    def __init__(self,path):       
        self.path=path
        self.data=None
        self.sentences=[]
        self.model=None
    

    def loaddata(self):
        with open(self.path,"rb") as f:     
            raw_data=f.read()
            detected_encoding=chardet.detect(raw_data)["encoding"]
            
        self.data=pd.read_csv(self.path,encoding=detected_encoding)["text"]
        self.text=self.data.str.cat(sep=" ")
    
    def process(self):
        self.text=self.text.lower()
        self.text=re.sub(r"[^a-zA-Z'\s]","",self.text)
        self.text=re.sub(r"\s+"," ",self.text).strip()
        all_sentences=self.text.split(".")      #将整个文本以句号作为分割,创建一个以句为基本单位的列表
        for sentence in all_sentences:
            sentence=re.sub(r"[^a-zA-Z'\s]","",sentence)        
            sentence=re.sub(r"\s+"," ",sentence).strip()
            sentence=sentence.split()       #每一句再次分割为一个列表
            sentence=[word for word in sentence]
            if sentence:        #防止sentence列表是空列表
                self.sentences.append(sentence)
                
            
    def train(self,vector_size=300,window=10,min_count=5,workers=4):     #训练模型
        self.model=Word2Vec(
                            sentences=self.sentences,      
                            #sentences参数接受的是一个二层嵌套的列表,其中每个元素是一个列表,这个子列表包含了一个句子中的单词
                            #结构类似于[["word1","word2","word3"],["word4","word5"],...]
                            vector_size=vector_size,        #词向量的维度
                            window=window,      #词关联范围
                            min_count=min_count,        #词出现的最小次数
                            workers=workers,        
                            )
        
        
    def savemodel(self):
        self.model.save("my_word2vec.model")
        
        
    def get_similar_words(self,word,topn=10):
        try:
            similar_words=self.model.wv.most_similar(word,topn=topn)
            return similar_words
        except KeyError:
            print(f"词汇{word}不在词汇列表中")
            return None
            

def main():
    w2v=my_word2vec("imdb_tr.csv")
    w2v.loaddata()
    w2v.process()
    w2v.train()
    w2v.savemodel()
    similar_words=w2v.get_similar_words("movie")
    print(similar_words)
    

if __name__=="__main__":
    main()