from datasets import load_dataset 
import os
import sys


current_path=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.abspath(os.path.join(current_path,".."))
sys.path.append(project_root)
from five_config import config


class Download_data():
    
    def __init__(self,dataset_name=config.DATA_NAME,path=config.DATA_DOWNLOAD_PATH):
        
        self.dataset_name=dataset_name
        self.path=path
        
        
    def download(self):
        
        current_dir=os.path.dirname(os.path.abspath(__file__))
        data_dir=os.path.join(current_dir,"..",self.path)
        
        data_train_dir=os.path.join(data_dir,"train")
        data_validation_dir=os.path.join(data_dir,"validation")
        os.makedirs(data_train_dir,exist_ok=True)
        os.makedirs(data_validation_dir,exist_ok=True)
        
        dataset=load_dataset(self.dataset_name)
        
        print(dataset)
        print(dataset["train"][0])
        
        dataset["train"].save_to_disk(data_train_dir)
        dataset["validation"].save_to_disk(data_validation_dir)
        
        return dataset["train"],dataset["validation"]