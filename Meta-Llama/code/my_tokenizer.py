from transformers import AutoTokenizer
import os
import sys
from torch.utils.data import DataLoader


current_path=os.path.dirname(os.path.abspath(__file__))
porject_root=os.path.join(current_path,"..")
sys.path.append(porject_root)


from config import config


class MyTokenizer():
    
    def __init__(self):
        
        self.model_name=config.MODEL_NAME
        self.tokenizer_path=config.TOKENIZER_PATH
        self.tokenizer=None
        
        
    def download(self):
        
        current_path=os.path.dirname(os.path.abspath(__file__))
        cache_dir=os.path.join(current_path,"..",self.tokenizer_path)
        
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,cache_dir=cache_dir,use_fast=True,trust_remote_code=True)
        #如果使用的是huggingface上面的一些特殊模型(例如Meta-Llama),它们的tokenizer类不是标准库内置的,因此需要自定义
        
        print(self.tokenizer.all_special_tokens)
        #查看tokenizer自带的特殊词
        
        if self.tokenizer.pad_token is None:       
            self.tokenizer.pad_token=self.tokenizer.eos_token
        #由于llama模型的tokenizer没有pad_token,因此tokenizer不知道用什么token进行补齐,因此需要人为设置一个pad_token
        #llama模型的tokenizer自带一个eos_token,用于表示结束,因此我们可以将eos_token作为末尾填充的token标识
        
        if self.tokenizer.padding_side=="left":     #有些模型的tokenizer填充(pad)发生于左侧,可以自行设置为右侧
            self.tokenizer.padding_side="right"     
        
        return self.tokenizer
        
    
    def tokenize_dataset(self,dataset):
    
        def tokenize_example(example):
            
            prompt=example["instruction"]       #由于"instruction"中已经包含中文提示词,因此这里不需要再写提示词了
            response=example["response"]
            
            input=prompt+response       
            #此处将拼接数据作为训练模型的输入,和bert有些差异
            #bert是Encoder-only模型,其双向注意力使得编码每个token时能看到左右两侧的所有词,因此可以用来理解上下文本以及语句关系
            #llama是Decoder-only自回归语言模型,其工作方式是根据一个token预测下一个token,其主要能力是生成式能力
            #因此将指令+文本+摘要拼接在一起作为输入,就可以使得模型逐渐学习根据文本生成摘要、根据摘要的开头生成后续摘要内容(本质就是续写token的能力)
            #训练时我们将拼接数据作为输入,但是模型后续应用时我们只需要提供instruction
            
            tokenized_input=self.tokenizer(
                                           input,
                                           padding="max_length",      
                                           truncation=True,       #对超过的长度进行截断
                                           max_length=config.MAX_LEN,
                                          )
            
            prompt_len=len(self.tokenizer(prompt,padding=False,truncation=True,max_length=config.MAX_LEN)["input_ids"])    
            
            temp_labels=prompt_len*[-100]+tokenized_input["input_ids"][prompt_len:]
            labels=[-100 if t==self.tokenizer.pad_token_id else t for t in temp_labels]
            #但是我们最终只需要模型正确生成摘要,因此我们只关心模型生成的摘要部分的损失
            #将填补的地方设置为-100,在后续计算loss时会自动忽略-100的位置(CrossEntropyLoss在计算loss时会自动忽略-100的位置)
            
            return {"input_ids":tokenized_input["input_ids"],"attention_mask":tokenized_input["attention_mask"],"labels":labels}       #squeeze去除第0维度
        
        dataset=dataset.map(tokenize_example,remove_columns=dataset.column_names)       #去除原本多余的列名
        
        return dataset
    
    
    def dataloader(self,dataset):
        
        dataset.set_format(type="torch",columns=["input_ids","attention_mask","labels"])
        #将dataset的格式修改为torch类型,原本为list类型
        
        return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True)
    
    
    def show_dataloader(self,dataloader):       #展示数据包装后的基本信息              
        
        for batch in dataloader:
            
            input_ids=batch.get("input_ids",None)
            attention_mask=batch.get("attention_mask",None)
            labels=batch.get("labels",None)
            
            print("input_ids---shape:",input_ids.shape if input_ids is not None else None)
            print("attention_mask---shape:",attention_mask.shape if attention_mask is not None else None)
            print("labels---shape:",labels.shape if labels is not None else None)
            
            print("input_ids---tensor:",input_ids[0])
            print("attention_mask---tensor",attention_mask[0])
            print("labels---tensor",labels[0])
            
            break