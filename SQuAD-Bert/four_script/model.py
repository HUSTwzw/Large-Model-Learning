from transformers import AutoModel
from torch import nn as nn



class model(nn.Module):
    
    
    def __init__(self,model_name,model_path,dropout):
        
        super().__init__()
        self.bert=AutoModel.from_pretrained(model_name,cache_dir=model_path)
        self.dropout=nn.Dropout(dropout)
        self.linear=nn.Linear(self.bert.config.hidden_size,2)
        #如果只有一个线性层就不能随意设置隐藏层维度,必须确保隐藏层维度是bert模型的输出维度
        
        
    def forward(self,input_ids,token_type_ids,attention_mask):
        
        output=self.bert(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        last_hidden_output=output.last_hidden_state      
        #此处获取最后一层隐藏层输出,形状为(batch_size,seq_length,hidden_size)
        #注意:之前的情感分类任务中,使用output.last_hidden_state[:0:],是因为bert的序列中第0个token就是"cls",因此只需要选取"cls"的对应输出
        #    但是这次是问答任务,因此需要获取最后一层隐藏层输出的全部信息
        last_hidden_output=self.dropout(last_hidden_output)
        logits=self.linear(last_hidden_output)
        
        start_logits,end_logits=logits.split(1,dim=-1)      #将(batch_size,seq_length,2)分割成start和end两部分
        start_logits=start_logits.squeeze(-1)       #消除最后一维度,变成(batch_size,seq_length)
        end_logits=end_logits.squeeze(-1)       #消除最后一维度,变成(batch_size,seq_length)
        
        return start_logits,end_logits