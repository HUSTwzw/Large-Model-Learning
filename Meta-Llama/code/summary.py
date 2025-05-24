import sys
import os
from accelerate import Accelerator


current_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.join(current_path,"..")


from config import config


class MySummarizer():
    
    def __init__(self,tokenizer,model):
        
        self.tokenizer=tokenizer
        self.model=model
        self.accelerator=Accelerator()
        self.tokenizer,self.model=self.accelerator.prepare(self.tokenizer,self.model)
        
    
    def summarize(self,text):

        prompt=f"请为下面的文章生成摘要\n{text}"
        
        input=self.tokenizer(
                             prompt,
                             padding="max_length",
                             truncation=True,
                             max_length=config.MAX_LEN,
                             return_tensors="pt"
                            )
        
        input_ids=input["input_ids"].to(self.accelerator.device)        #由于input没有被accelerator包装,因此可能出现模型在"gpu",输入在"cpu",从而引发不匹配的错误,因此需要手动将input放在模型所在位置
        attention_mask=input["attention_mask"].to(self.accelerator.device)
        
        output_ids=self.model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=config.OUTPUT_MAX_LEN)        #此处返回的是(1,seq_len)形状的token_ids
        summary=self.tokenizer.decode(output_ids[0],skip_special_tokens=True)
       
        return summary