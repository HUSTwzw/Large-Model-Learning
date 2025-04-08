#复现注意力机制(单头注意力机制与多头注意力机制)


import torch
from torch import nn


torch.manual_seed(22)       
#设置随机数种子
#设置随机种子是为了在每次运行中获得一致的初始化结果，便于调试和实验对比。


class MySelfAttention(nn.Module):
    def __init__(self,embedding_dim):
        super(MySelfAttention,self).__init__()
        self.embedding_dim=embedding_dim
        self.Wq=nn.Linear(embedding_dim,embedding_dim)        #创建Wq矩阵
        self.Wk=nn.Linear(embedding_dim,embedding_dim)        #创建Wk矩阵
        self.Wv=nn.Linear(embedding_dim,embedding_dim)        #创建Wv矩阵
        
        
    def forward(self,x):        #x的形状为(batch_size,seq-length,embedding_dim)
        batch_size,seq_length,embedding_dim=x.size()
        
        Q=self.Wq(x)        #形状为(batch_size,seq_length,embedding_dim)
        K=self.Wk(x)        #形状为(batch_size,seq_length,embedding_dim)
        V=self.Wv(x)        #形状为(batch_size,seq_length,embedding_dim)
        
        attention_scores=torch.matmul(Q,K.transpose(1,2))      
        #计算向量之间的点积
        #K.transpose(1,2)交换一二维(维度从0开始),形状为(batch_size,embedding_dim,seq_length)
        #对于二维矩阵,matmul相当于Q@K.T,得到的形状为(seq_length,seq_length)
        #对于三维及以上的张量(例如当前情况),matmul会自动对最后两维执行矩阵乘法,并对前面的维度执行广播,因此得到形状为(batch_size,seq_length,seq_length)
        #点积表示相似度,由于计算本质是Q@K.T,因此(batch_size,seq_length,seq_length)矩阵中每一个元素(batch_size,i,j)表示i的q和j的k的相似度
        
        attention_scores=attention_scores/(embedding_dim**0.5)        #对点积相似度进行缩放,防止数值过大(通常除以嵌入维度的平方根)
        
        attention_weights=nn.functional.softmax(attention_scores,dim=2)      
        #在第二维度上进行softmax归一化
        #每个查询(Q[i])与所有键(K[j])的相似度对应的权重之和为1
        #这个矩阵展示了所有键在每一个查询中的权重,因此被称为attention_weights
        
        O=torch.matmul(attention_weights,V)
        
        #Q是你要查询的内容,表示你对某个位置的兴趣
        #K是提供“信息”的键,表示其他位置的信息
        #V是实际的值(数据),表示你想要提取的信息
        #每个查询(Q)会与所有的键(K)进行比较,得到一个相似度,然后这些相似度会被转化为权重,最终用这些权重对值(V)进行加权求和,得到该查询的输出
        
        return O        #返回输出
    
    
class MyMultiHeadSelfAttention(nn.Module):
    def __init__(self,batch_size,seq_length,embedding_dim,head_num):
        super(MyMultiHeadSelfAttention,self).__init__()
        self.batch_size=batch_size
        self.seq_length=seq_length
        self.embedding_dim=embedding_dim
        self.head_num=head_num
        self.head_dim=embedding_dim//head_num       #一般需要embedding_dim和head_num是整数倍关系
        
        self.Wq=nn.Linear(embedding_dim,embedding_dim)
        self.Wk=nn.Linear(embedding_dim,embedding_dim)
        self.Wv=nn.Linear(embedding_dim,embedding_dim)
        
        self.Wo=nn.Linear(embedding_dim,embedding_dim)
        
        
    def forward(self,x):
        Q=self.Wq(x)
        K=self.Wk(x)
        V=self.Wv(x)
        
        Q=Q.view(self.batch_size,self.seq_length,self.head_num,self.head_dim)       #将Q形状改为(batch_size,seq_length,head_num,head_dim),每个头分配一个形状为(batch_size,seq_length,head_dim)形状的矩阵
        K=K.view(self.batch_size,self.seq_length,self.head_num,self.head_dim)       #将K形状改为(batch_size,seq_length,head_num,head_dim),每个头分配一个形状为(batch_size,seq_length,head_dim)形状的矩阵
        V=V.view(self.batch_size,self.seq_length,self.head_num,self.head_dim)       #将V形状改为(batch_size,seq_length,head_num,head_dim),每个头分配一个形状为(batch_size,seq_length,head_dim)形状的矩阵
        #将QKV改为多头形式
        
        Q=Q.permute(0,2,1,3)        #将Q形状改为(batch_size,head_num,seq_length,head_dim)
        K=K.permute(0,2,1,3)        #将K形状改为(batch_size,head_num,seq_length,head_dim)
        V=V.permute(0,2,1,3)        #将V形状改为(batch_size,head_num,seq_length,head_dim)
        #使用permute调整维度顺序,便于并行计算多头        
                
        attention_scores=torch.matmul(Q,K.transpose(2,3))       
        #attention_scores形状为(batch_size,head_num,seq_length,seq_length)
        
        attention_scores=attention_scores/(self.head_dim**0.5)      #相似度缩放
        
        attention_weights=nn.functional.softmax(attention_scores,dim=-1)        #生成权重矩阵
        #在最后一个维度(即每个query对所有key)上进行softmax归一化
        #得到注意力权重矩阵,形状为(batch_size,head_num,seq_length,seq_length)
        #每一行表示一个query对所有位置key的权重分布
        
        O=torch.matmul(attention_weights,V)
        #O形状为(batch_size,head_num,seq_length,head_dim)
        
        O=O.permute(0,2,1,3)
        #O形状改为(batch_size,seq_length,head_num,head_dim)
        
        O=O.contiguous().view(self.batch_size,self.seq_length,self.embedding_dim)
        #view函数的使用需要保证张量是连续的
        #之前使用permute虽然表面改变张亮形状(维度顺序改变),但是不会改变底层内存布局,导致张量形状与实际内存分布不一致(即非连续)
        #contiguous()返回一个内存布局与当前形状一致(连续)的张量,以便后续使用view
        
        O=self.Wo(O)
        #通过Wo进行线性变换
        #多头注意力的输出再次通过Wo映射回原始的embedding_dim空间
        
        return O        