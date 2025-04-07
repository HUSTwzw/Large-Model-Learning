#LSTM随机生成电影评论文本


import torch
from torch import nn,optim
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset,DataLoader
import chardet
import re
import numpy as np
import argparse


class Model(nn.Module):
    def __init__(self,dataset):
        super(Model,self).__init__()
        #super是一个内置函数,用于调用父类(nn.Module)的方法,通过调用父类,可以初始化神经网络的底层机制,否则子类(Model)无法使用父类(nn.Module)提供的功能
        self.embedding_dim=128      
        #嵌入层维度:实际上是将单词对应的向量降维成一个低维嵌入向量,不仅节省空间,同时通过后续学习也能体现单词之间相似性以及语义特征  
        #嵌入层会不断学习优化:通过直接找到输入的单词对应的嵌入向量,优化这些输入单词对应的嵌入向量(整个操作称为查表)

        self.lstm_hidden=128      
        #LSTM隐藏层的维度:LSTM的隐藏层维度不一定与嵌入层维度相同,虽然经常令embedding_dim==lstm_hidden,但是主要为了简化运算,在面临更复杂的问题时,通常会使得lstm_hidden更大
        #lstm的处理过程可以理解为将嵌入层作为lstm的输入,经过输入门、遗忘门、输出门、细胞状态更新,最终的隐藏层输出为lstm_hidden维度
        #嵌入层输出为(batch_size,seq_length,embedding_dim),经过输入门、遗忘门、输出门、细胞状态矩阵操作,最终的隐藏层输出为(batch_size,seq_length,lstm_hidden)
        #输入门、遗忘门、输出门、细胞状态操作矩阵形状为(embedding_dim+lstm_hidden,lstm_hidden),因为输入数据要和前一个时间步的隐藏层输出拼接到一起

        self.lstm_layers=3      
        #堆叠的LSTM层数:表示每一批次的输入数据都经过三层LSTM层,最终实现隐藏层输出
        
        vocab_num=len(dataset.uniq_words)     #唯一词的数量
        
        self.embedding=nn.Embedding(        
                                    num_embeddings=vocab_num,        #num_embeddings接收唯一词的数量
                                    embedding_dim=self.embedding_dim        #embedding_dim接受嵌入层的维度
                                   )        
        #创建一个nn.Embedding类的实例
        
        self.lstm=nn.LSTM(
                          input_size=self.embedding_dim,        #输入层维度:应与嵌入层输出维度一致
                          hidden_size=self.lstm_hidden,     
                          #hidden_size是隐藏层维度,决定隐藏状态大小,也是LSTM每个时间步输出的维度
                          #每个时间步隐藏层输出(batch_size,1,lstm_hidden),lstm处理整个序列后生成每个时间步合成了(batch_size,seq_length,lstm_hidden)
                          num_layers=self.lstm_layers,       #LSTM隐藏层的数量
                          dropout=0.2,       #dropout是一种正则化技术,在训练过程中,dropout会随机将一部分神经元的输出设置为零,从而防止神经元之间形成过于紧密的依赖关系,防止过拟合(dropout只在lstm_layers>1时才生效,最后一层不使用dropout)
                          batch_first=True      #PyTorch的nn.LSTM默认输入维度顺序为(seq_length,batch_size,input_size),而embed形状为(batch_size,seq_length,embedding_dim),因此需要batch_first
                         )
        #创建一个nn.LSTM类的实例
        
        self.fc=nn.Linear(
                          in_features=self.lstm_hidden,     #in_features接收输入维度
                          out_features=vocab_num        #out_features接收输出维度
                         )       
        #全连接:将LSTM的每个时间步输出(隐藏层)利用线性操作从lstm_hidden维度映射到词汇表vocab_num维度

        
    def forward(self,x,prev_state):     
        #x是输入数据,形状为(batch_size,seq_length):x矩阵中的元素都是当前序列中单词对应的词汇表的序号        
        #prev_state是一个元组,包含两个张量,分别表示隐藏状态state_h和细胞状态state_c(表示的是每个时间步的隐藏状态和细胞状态,而不是整个序列的输出)
        #其中state_h就是lstm隐藏层的输出向量,形状为(lstm_layers,batch_size,lstm_hidden),负责传递短期记忆;state_c负责捕捉长期依赖关系,形状也为(lstm_layers,batch_size,lstm_hidden)
        
        embed=self.embedding(x)     
        #将输入转换为嵌入词向量
        #此处将形状为(batch_size,seq_length)的x变成(batch_size,seq_length,embedding_dim)的embed
        
        output,state=self.lstm(embed,prev_state)        #将嵌入向量输入LSTM,得到输出和新状态
        #对于多层LSTM,这里的output是LSTM最后一层隐藏状态,是一个三维张量,形状为(batch_size,seq_length,lstm_hidden)
        #state是一个元组,包含两个张量,state_h是所有LSTM层在最后一个时间步的隐藏状态,形状为(lstm_layers,batch_size,lstm_hidden),state_c是所有LSTM层在最后一个时间步的细胞状态,形状为(lstm_layers,batch_size,lstm_hidden)
        #output是最后一层LSTM在所有时间步的隐藏状态集合(只保留最后一层输出),state包含所有层在最后一个时间步的状态（包含多层信息）
        
        logits=self.fc(output)      
        #将LSTM输出映射到词汇表大小,得到预测结果,形状为(batch_size,seq_length,vocab_num)
        
        return logits,state     #返回预测结果和新状态
    
    
    def init_state(self,batch_size):        #初始化state_h与state_c
        return (torch.zeros(self.lstm_layers,batch_size,self.lstm_hidden),torch.zeros(self.lstm_layers,batch_size,self.lstm_hidden))        
        #返回两个全零张量(state_h与state_c形状均为(lstm_layers,batch_size,lstm_hidden))
    
    
class Dataset(Dataset):
    def __init__(self,args,path,line):
        self.args=args
        self.path=path
        self.line=line
        self.wordlist=self.loadwords(path,line)
        self.uniq_words=self.get_uniq_words()
        self.index_to_words={index:word for index,word in enumerate(self.uniq_words)}
        self.word_to_index={word:index for index,word in enumerate(self.uniq_words)}
        self.wordlist_to_indexs=[self.word_to_index[word] for word in self.wordlist]
        
        
    def loadwords(self,path,line):
        with open(path,"rb") as f:
            raw_data=f.read()
            detected_encoding=chardet.detect(raw_data)["encoding"]      #检测文件格式
        data=pd.read_csv(path,encoding=detected_encoding)       #读取文件数据
        
        data=data[:line]        #提取一定行数的数据
        text=data["text"].str.cat(sep=" ")      #将所有数据连接成一个字符串
        text=re.sub(r"[^a-zA-Z]"," ",text)      #去除非字母
        text=re.sub(r"\s+"," ",text)        #去除多余空格,确保单词间仅有一个空格
        text=text.lower()       #将单词小写
        wordlist=[word for word in text.split(" ") if word]      #去除空字符
        return wordlist
    
    
    def get_uniq_words(self):
        word_counts=Counter(self.wordlist)
        return sorted(word_counts,key=word_counts.get,reverse=True)     #根据词出现次数逆序排序
    
    
    def __len__(self):      #计算有效样本数,即总长度-序列长度
        return len(self.wordlist_to_indexs)-self.args.seq_length     #sequence_length是输入序列的长度,也是目标输出序列的长度
        
        
    def __getitem__(self,index):
        return (
                torch.tensor(self.wordlist_to_indexs[index:index+self.args.seq_length]),      #输入序列:从index开始的seq_length个词
                torch.tensor(self.wordlist_to_indexs[index+1:index+1+self.args.seq_length])       #目标序列:输入序列后移一位,从index+1开始的seq_length个词
               )
        
        
class Train():
    def __init__(self,dataset,model,args):
        model.train()       #将模型改成训练模式     
        
        dataloader=DataLoader(dataset,batch_size=args.batch_size)       
        #将dataset分批次加载
        #根据Dataset类中getitem方法,可以得知DataLoader将数据(输入数据x和输出数据y)包装成(batch_size,seq_length)的形状
        
        criterion=nn.CrossEntropyLoss()     #定义损失函数为交叉熵损失
        
        optimizer=optim.Adam(model.parameters(),lr=0.001)       #parameters方法返回一个生成器,遍历所有需要梯度的参数(张量)      Adam是常用的优化器
        
        for epoch in range(args.max_epochs):        #开始一个循环,循环次数由训练轮数max_epochs决定,对相同的数据进行重复训练
            
            for batch,(x,y) in enumerate(dataloader):       #遍历dataloader实例的每一个批次
                
                batch_size=x.size(0)
                state_h,state_c=model.init_state(batch_size)       
                #将batch_size传入初始化函数,初始化隐藏层状态和细胞状态
                #注意:此项目是随机生成电影评论文本,由于数据是多人的电影评论,每个评论相对独立,因此隐藏状态在不同batch中不需要传递,因此每一个batch都重新初始化一个state_h与state_c
                
                optimizer.zero_grad()       
                #将当前batch的梯度清零,确保当前batch计算得到的梯度独立,不受之前batch的影响
                #注意:将当前batch的梯度清零不会导致后续optimizer.step()无法更新参数,反而可以防止不同batch间梯度堆积,防止后续batch的参数更新错误     
                
                y_pred,(state_h,state_c)=model(x,(state_h,state_c))     #调用模型进行前向传播(自动调用nn.Module中forward函数)
                
                loss=criterion(y_pred.transpose(1,2),y)     
                #y的维度是(batch_size,seq_length),y_pred的维度是(batch_size,seq_length,vocab_num)
                #而nn.CrossEntropyLoss所接收的形状是(batch_size,vocab_num,seq_length),因此需要交换一维和二维(维度从0开始)
                
                state_h=state_h.detach()
                state_c=state_c.detach()
                #detach是将隐藏状态state_h与细胞状态state_c从计算图中分离开,反向传播时,梯度不会传播到之前的时间步
                #形状为(batch_size,seq_length)的x经过嵌入层变成(batch_size,seq_length,embedding_dim)作为input,而每一个时间步相当于处理(batch_size,1,embedding_dim)
                #因此detach将反向传播限制在当前的(batch_size,seq_length,embedding_dim),但是state_c的存在保留的之前的"记忆",因此没有传播到之前的时间步并无影响
                
                loss.backward()     #进行反向传播(此时自动计算出了当前batch的梯度)
                optimizer.step()        #使用优化器更新模型参数,最小化损失
                print({"epoch":epoch,"batch":batch,"loss":loss.item()})    
                
                
def predect(dataset,model,text,next_words=100):     
    model.eval()        #模型设置为评估模式
    
    words=text.split(" ")
    state_h,state_c=model.init_state(1)
    
    for i in range(next_words):     #next_words表示要根据已有text连续生成的单词的数量
        x=torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])     #次数x的形状为(1,seq_length)
        y_pred,(state_h,state_c)=model(x,(state_h,state_c))
        last_word_logits=y_pred[0][-1]      
        #y_pred是一个三维张量,形状为(batch_size,seq_length,vocab_num)     
        #由于只处理一个样本,batch_size为1,而seq_length是单词x输入的单词个数(大小等于text长度),因此last_word_logits的形状为(vocab_num)
        
        p=torch.nn.functional.softmax(last_word_logits,dim=0).detach().numpy()      
        #调用softmax函数计算不同单词的可能概率,detach可以将张量从计算图中分离,阻止梯度回传
        #使用numpy构建长度为vocab_num的数组,展现不同单词的可能概率
           
        word_index=np.random.choice(np.arange(len(last_word_logits)),p=p)
        #此处给出不同序号的对应频率,然后根据频率随机选择(不直接选择最大频率的词序号是为了增强输出的灵活性)
        
        words.append(dataset.index_to_words[word_index])
        #将新的输出添加到列表中,作为下一次的输入
    
    return words
                    
                    
parser=argparse.ArgumentParser()
parser.add_argument("--max-epochs",type=int,default=10)     #确定循环次数
parser.add_argument("--batch-size",type=int,default=256)        #确定一个batch的大小
parser.add_argument("--seq-length",type=int,default=16)     #确定seq_length的大小
args=parser.parse_args()


dataset=Dataset(args,"imdb_tr.csv",1000)
model=Model(dataset)
train=Train(dataset,model,args)
print(predect(dataset,model,"think this movie"))