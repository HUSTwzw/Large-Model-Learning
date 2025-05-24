import sys
import os
from accelerate import Accelerator
import torch
from transformers import get_scheduler
from tqdm import tqdm


from config import config


current_path=os.path.dirname(os.path.abspath(__file__))
root_path=os.path.join(current_path,"..")
sys.path.append(root_path)


class MyTrainer():
    
    def __init__(self,train_dataloader,tokenizer,model):
        
        self.train_dataloader=train_dataloader
        self.tokenizer=tokenizer
        self.model=model
        self.accelerator=Accelerator()      #自动将模型、数据和优化器等迁移到最优的设备
        self.optimizer=None
        self.lr_scheduler=None
        
        
    def setup_optimizer_scheduler(self):
        
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config.LR)
        self.lr_scheduler=get_scheduler(
                                        name="linear",
                                        optimizer=self.optimizer,       #学习率变化器会将optimizer的最初学习率作为学习率变化器的峰值
                                        num_warmup_steps=100,       #热身:通过100步达到config.LR
                                        num_training_steps=len(self.train_dataloader)*config.NUM_EPOCHS,        #学习率变化总次数:每一个batch都会更新一次学习率,因此学习率应该是batch批次个数*循环训练次数
                                       )
        
        self.model,self.optimizer,self.train_dataloader=self.accelerator.prepare(self.model,self.optimizer,self.train_dataloader)
        
    
    def train(self):
        
        self.model.train()
        
        for epoch in range(config.NUM_EPOCHS):
            
            total_loss=0
            total_step=0
            average_loss=0
            
            for step,batch in enumerate(tqdm(self.train_dataloader,desc="Training")):
                
                total_step+=1
                
                output=self.model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])       #通过将labels作为输入huggingface可以直接损失
                
                loss=output.loss      #huggingface的transformer库已经封装好了,可以直接计算损失
                total_loss+=loss.item()
                
                self.accelerator.backward(loss)     #根据损失,利用accelerator反向传播
                self.optimizer.step()       #更新参数
                self.lr_scheduler.step()        #更新学习率
                
                self.optimizer.zero_grad()      #清空梯度(将模型所有参数的梯度清空)
                
                print(f"Batch {step}:{loss.item():.4f}")
            
            average_loss=total_loss/total_step
            
            print(f"Epoch {epoch}:{average_loss:.4f}")
                
                
    def save_lora(self):
        
        current_path=os.path.dirname(os.path.abspath(__file__))
        lora_path=os.path.join(current_path,"..",config.LORA_PATH)
        tokenizer_path=os.path.join(current_path,"..",config.NEW_TOKENIZER_PATH)
        
        unwrapped_model=self.accelerator.unwrap_model(self.model)       #获取原始、非封装模型(去除accelerator包装)
        unwrapped_model.save_pretrained(lora_path)       #将LoRA训练后的参数保存在指定位置
        self.tokenizer.save_pretrained(tokenizer_path)