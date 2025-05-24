import sys 
import os
from tqdm import tqdm
from accelerate import Accelerator
import torch
import evaluate


current_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.join(current_path,"..")
sys.path.append(root_path)


from config import config


class MyEvaluator():
    
    def __init__(self,test_dataloader,tokenizer,model):

        self.test_dataloader=test_dataloader
        self.tokenizer=tokenizer
        self.model=model
        self.accelerator=Accelerator()
        self.test_dataloader,self.tokenizer,self.model=self.accelerator.prepare(self.test_dataloader,self.tokenizer,self.model)
        self.rouge=evaluate.load("rouge")       #是一组用于自动评估文本摘要质量的指标,通过比较机器生成文本(候选摘要)和人工参考文本(真实摘要)的重叠程度来衡量生成文本的质量  
    

    def evaluate(self):
        
        self.model.eval()
        
        total_step=0
        total_loss=0
        average_loss=0
        
        generated_text=[]
        target_text=[]
        
        with torch.no_grad():
        
            for step,batch in enumerate(tqdm(self.test_dataloader,desc="EVALUATION")):
                
                total_step+=1
                
                output=self.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
                
                loss=output.loss
                total_loss+=loss.item()
                
                generated_ids=self.model.generate(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],max_new_tokens=config.OUTPUT_MAX_LEN)     #以batch为基本规模生成token_id
                decoded_text=self.tokenizer.batch_decode(generated_ids,skip_special_tokens=True)        #以batch为基本单位将token_id转化成列表(列表中有batch_size个字符串文本,即batch_size个文本摘要)
                generated_text.extend(decoded_text)     #将列表的元素融入一个列表
                
                temp_labels=batch["labels"].clone()     #克隆一份可以防止对原本的tensor进行修改
                temp_labels[temp_labels==-100]=self.tokenizer.pad_token_id      #将原本的-100位置设置为pad_token_id
                decoded_target=self.tokenizer.batch_decode(temp_labels,skip_special_tokens=True)
                target_text.extend(decoded_target)
                
                
            average_loss=total_loss/total_step   
            
            print(f"TESTING---AverageLoss:{average_loss:.4f}")
            
            result=self.rouge.compute(predictions=generated_text,references=target_text)
            
            print(f"ROUGE-1 {result['rouge1']:.4f}")
            print(f"ROUGE-2 {result['rouge2']:.4f}")
            print(f"ROUGE-L {result['rougeL']:.4f}")