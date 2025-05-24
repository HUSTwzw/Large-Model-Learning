from transformers import AutoModelForCausalLM
import sys
import os
import torch.nn as nn
from peft import LoraConfig,TaskType,get_peft_model


current_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.join(current_path,"..")
sys.path.append(root_path)


from config import config


class MyModel():
    
    def __init__(self):
        
        self.model_name=config.MODEL_NAME
        self.model_path=config.MODEL_PATH
    
    
    def download(self):
        
        current_path=os.path.dirname(os.path.abspath(__file__))
        cache_dir=os.path.join(current_path,"..",self.model_path)
        self.base_model=AutoModelForCausalLM.from_pretrained(self.model_name,cache_dir=cache_dir,trust_remote_code=True,device_map="auto")
        
        return self.base_model
    
    
    def apply_lora(self):

        lora_config=LoraConfig(
                               r=config.R,
                               lora_alpha=config.LORA_ALPHA,
                               lora_dropout=config.LORA_DROPOUT,
                               bias="none",     #bias="none"表示不对矩阵偏置项使用LoRA(属于常见操作)
                               target_modules=config.TARGET_MODULES,
                               task_type=TaskType.CAUSAL_LM     #对于Llama模型通常使用CAUSAL_LM
                              )
        
        self.model=get_peft_model(self.base_model,lora_config)
        
        return self.model