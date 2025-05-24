from datasets import load_dataset
import os
import sys


current_path=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.join(current_path,"..")
sys.path.append(project_root)


from config import config


class MyDataset():
    
    
    def __init__(self):
        
        self.dataset_name=config.DATA_NAME
        self.dataset_path=config.DATA_PATH
        
    
    def download(self):
        
        current_path=os.path.dirname(os.path.abspath(__file__))
        cache_dir=os.path.join(current_path,"..",self.dataset_path)

        dataset=load_dataset(self.dataset_name,cache_dir=cache_dir)
        
        print(dataset)
        print(dataset["train"].column_names)
        print(dataset["train"][0])
        
        return dataset
    
    
    def to_instruction_format(self,dataset):
        
        def format_example(example):
            
            return {
                    "instruction":f"请为下面的文章生成摘要:\n{example['document']}",
                    "response":example["summary"]
                   } 
        
        dataset=dataset.map(format_example,remove_columns=dataset.column_names)     
        #map函数默认将新字段添加到原有样本中,因此数据的原字段依然会保留(例如"document","summary","id")
        #如果想去除这些字段,只保留新字段,可以使用remove_columns去除原本的列
        
        return dataset