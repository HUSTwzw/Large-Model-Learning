import os
import sys
from tqdm import tqdm
import torch


current_path=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.join(current_path,"..")
sys.path.append(project_root)
from five_config import config


class train():
    
    
    def __init__(self,model,device,train_loader,validation_loader,criterion,optimizer,scheduler,epoches_num):
        
        self.model=model
        self.device=device
        self.train_loader=train_loader
        self.validation_loader=validation_loader
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.epoches_num=epoches_num
        
        
    def train_one_epoch(self):
        
        self.model.train()
        
        total_loss=0
        
        for batch_ids,batch in enumerate(tqdm(self.train_loader,desc="TRAINING")):
            
            print(batch)
            
            input_ids=batch[0].to(self.device)      #将数据放在指定的device上(cpu或者gpu),便于后续运算
            token_type_ids=batch[1].to(self.device)
            attention_mask=batch[2].to(self.device)
            start_position=batch[3].to(self.device)
            end_position=batch[4].to(self.device)
            
                        
            self.optimizer.zero_grad()
            
            start_logits,end_logits=self.model(input_ids,token_type_ids,attention_mask)
            
            start_loss=self.criterion(start_logits,start_position)
            end_loss=self.criterion(end_logits,end_position)
            loss=(start_loss+end_loss)/2
            loss.backward()
            self.optimizer.step()
            total_loss+=loss.item()
            
            print(f"BATCH_IDS:{batch_ids+1}/{len(self.train_loader)}---LOSS:{loss.item():.4f}")
            
            self.scheduler.step()       #此处选择的scheduler是warmup,会在每一个batch中更新学习率(学习率先热身递增,之后再逐渐减小)
            
        average_loss=total_loss/len(self.train_loader)
        
        return average_loss
    
        
    def evaluate_one_epoch(self):
        
        self.model.eval()

        total_loss=0
        total=0
        correct=0
        
        with torch.no_grad():
            for batch in tqdm(self.validation_loader,desc="TESTING"):
                
                input_ids=batch[0].to(self.device)  
                token_type_ids=batch[1].to(self.device)
                attention_mask=batch[2].to(self.device)
                start_position=batch[3].to(self.device)
                end_position=batch[4].to(self.device)
                
                start_logits,end_logits=self.model(input_ids,token_type_ids,attention_mask)

                start_loss=self.criterion(start_logits,start_position)
                end_loss=self.criterion(end_logits,end_position)
                loss=(start_loss+end_loss)/2
                total_loss+=loss.item()
                
                start=torch.argmax(start_logits,dim=1)
                end=torch.argmax(end_logits,dim=1)
                
                correct+=((start==start_position)&(end==end_position)).sum().item()     
                total+=input_ids.size(0)
                
        accuracy=correct/total
        
        return accuracy            
            
        
    def train_all_epoches(self):
        
        self.model.to(self.device)      #将模型移动到指定设备上(cpu或者gpu)    
        
        for epoch in range(self.epoches_num):
            
            average_loss=self.train_one_epoch()
            
            print(f"EPOCH:{epoch+1}/{self.epoches_num}---AVERAGE_LOSS:{average_loss:.4f}")
            
            accuracy=self.evaluate_one_epoch()
            
            print(f"EPOCH:{epoch+1}/{self.epoches_num}---ACCURACY:{accuracy:.4f}")

            
    def save(self,path):
        
        torch.save(self.model.state_dict(),path)