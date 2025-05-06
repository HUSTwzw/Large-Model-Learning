import os
import sys
import torch
from torch import nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


current_path=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.abspath(os.path.join(current_path,".."))
sys.path.append(project_root)
from five_config import config


from download_data import Download_data
from processing import process_data
from model import model
from train import train

if __name__=="__main__":

    main_path=os.path.dirname(os.path.abspath(__file__))

    download=Download_data(config.DATA_NAME)
    train_dataset,validation_dataset=download.download()

    process=process_data(config.MODEL_NAME,os.path.join(main_path,"..",config.MODEL_TOKENIZER_PATH),train_dataset,validation_dataset,config.MAX_LEN)
    tokenized_train,tokenized_validation=process.encode()
    train_loader,validation_loader=process.dataloader(tokenized_train,tokenized_validation)

    bert_model=model(config.MODEL_NAME,os.path.join(main_path,"..",config.MODEL_PATH),config.DROPOUT)

    total_steps=len(train_loader)*config.EPOCHES_NUM
    scheduler=get_linear_schedule_with_warmup(
                                              optimizer=AdamW(bert_model.parameters(),lr=config.LR),
                                              num_warmup_steps=int(0.1*total_steps),
                                              num_training_steps=total_steps,
                                             )

    trained_model=train(
                        bert_model,
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        train_loader,
                        validation_loader,
                        nn.CrossEntropyLoss(),
                        AdamW(bert_model.parameters(),lr=config.LR),
                        scheduler,
                        config.EPOCHES_NUM
                       )
    trained_model.train_all_epoches()
    trained_model.save(os.path.join(main_path,"..",config.SAVE_PATH))