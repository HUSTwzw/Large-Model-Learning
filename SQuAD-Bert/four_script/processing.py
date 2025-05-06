from transformers import AutoTokenizer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import os
import sys

current_path=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.abspath(os.path.join(current_path,".."))
sys.path.append(project_root)
from five_config import config


class process_data():
    
    
    def __init__(self,model_name,path,train_dataset,validation_dataset,max_len=config.MAX_LEN):
        
        self.model_name=model_name
        self.path=path
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,cache_dir=self.path)
        self.train_dataset=train_dataset
        self.validation_dataset=validation_dataset
        self.max_len=max_len

        print(self.tokenizer("hello python"))
    
    
    def encode_dataset(self, example):
       
        encoding = self.tokenizer(
                                  example["question"],
                                  example["context"],
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_len,
                                  return_offsets_mapping=True       #True表示能够显示每一个token对应的字符级别的位置区间
                                 )

        answer_start=example["answers"]["answer_start"][0]      #answer_start是字符级的序号
        answer=example["answers"]["text"][0]
        answer_end=answer_start+len(answer)-1
        start_token_index=end_token_index=0

        for i,(start,end) in enumerate(encoding["offset_mapping"]):     #尝试将字符级的序号转化为token级的序号
            if (start<=answer_start<end):
                start_token_index=i
            if (start<=answer_end<=end):
                end_token_index=i

        encoding["start_position"]=start_token_index
        encoding["end_position"]=end_token_index
        encoding.pop("offset_mapping")      #去除"offset_mapping"这一个键

        return encoding


    def encode(self):

        #由于map存在不明问题,此处依然使用手动循环
        train_features=[self.encode_dataset(example) for example in tqdm(self.train_dataset)]
        val_features=[self.encode_dataset(example) for example in tqdm(self.validation_dataset)]

        #构建新的Dataset对象
        tokenized_train=Dataset.from_dict({key:[f[key] for f in train_features] for key in train_features[0]})
        tokenized_validation=Dataset.from_dict({key:[f[key] for f in val_features] for key in val_features[0]})

        print(tokenized_train[0])       
        print(tokenized_validation[0])

        return tokenized_train,tokenized_validation
    
    
    def dataloader(self,tokenized_train,tokenized_validation):
        
        print(torch.tensor(tokenized_train["input_ids"]).shape)
        print(torch.tensor(tokenized_train["start_position"]).shape)
        
        train_dataset=TensorDataset(
                                    torch.tensor(tokenized_train["input_ids"]),
                                    torch.tensor(tokenized_train["token_type_ids"]),
                                    torch.tensor(tokenized_train["attention_mask"]),
                                    torch.tensor(tokenized_train["start_position"]),
                                    torch.tensor(tokenized_train["end_position"])
                                   )   
        
        validation_dataset=TensorDataset(
                                         torch.tensor(tokenized_validation["input_ids"]),
                                         torch.tensor(tokenized_validation["token_type_ids"]),
                                         torch.tensor(tokenized_validation["attention_mask"]),
                                         torch.tensor(tokenized_validation["start_position"]),
                                         torch.tensor(tokenized_validation["end_position"])
                                        )           
        
        train_loader=DataLoader(train_dataset,num_workers=config.NUM_WORKERS,batch_size=config.BATCH_SIZE,shuffle=True)            
        validation_loader=DataLoader(validation_dataset,num_workers=config.NUM_WORKERS,batch_size=config.BATCH_SIZE,shuffle=True)
        
        return train_loader,validation_loader