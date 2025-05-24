import os
import sys
from peft import PeftModel


current_path=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.join(current_path,"..")


sys.path.append(project_root)


from config import config
from build_dataset import MyDataset
from my_tokenizer import MyTokenizer
from my_model import MyModel
from train import MyTrainer
from evaluating import MyEvaluator
from summary import MySummarizer


if __name__=="__main__":
    
    #处理数据格式:llama需要特殊提示词(用于告知模型任务)
    dataset_=MyDataset()
    dataset=dataset_.download()
    train_dataset=dataset_.to_instruction_format(dataset["train"])
    test_dataset=dataset_.to_instruction_format(dataset["test"])
    
    print(train_dataset[0])
    print(test_dataset[0])
    
    #将数据处理为训练llama的基本格式
    tokenizer_=MyTokenizer()
    tokenizer=tokenizer_.download()
    train_dataset=tokenizer_.tokenize_dataset(train_dataset)
    test_dataset=tokenizer_.tokenize_dataset(test_dataset)
    
    print(train_dataset[0])
    print(train_dataset[1])
    
    train_dataloader=tokenizer_.dataloader(train_dataset)
    test_dataloader=tokenizer_.dataloader(test_dataset)
    
    tokenizer_.show_dataloader(train_dataloader)
    tokenizer_.show_dataloader(test_dataloader)   
    
    #加载模型
    model_=MyModel()
    base_model=model_.download() 
    model=model_.apply_lora()    
    
    #训练LoRA参数
    train_=MyTrainer(train_dataloader,tokenizer,model)
    train_.setup_optimizer_scheduler()
    train_.train()
    train_.save_lora()
    
    #评估训练结果
    final_lora_path=os.path.join(project_root,config.LORA_PATH)
    trained_model=PeftModel.from_pretrained(base_model,final_lora_path)
    evaluate_=MyEvaluator(test_dataloader,tokenizer,trained_model)
    evaluate_.evaluate()
    
    #展示输出
    text=input("请输入文本")
    summary_=MySummarizer(tokenizer,trained_model)
    summary=summary_.summarize(text)
    print(summary)