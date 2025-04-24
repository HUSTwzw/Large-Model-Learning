#利用HuggingFace中的Bert模型,结合imdb电影评论,训练一个情感分类模型


import torch
from torch.utils.data import DataLoader,TensorDataset
from torch import nn
import transformers     #HuggingFace制作的库,用于加载模型以及分词器
from transformers import AutoTokenizer,AutoModel
from torch.optim import AdamW      #HuggingFace推荐的优化器之一
import datasets     #HuggingFace制作的库,用于下载训练数据与测试数据
from datasets import load_dataset
import os
from tqdm import tqdm




os.chdir("C:/Users/HUSTwzw/Desktop/Bert-EmotionClassification")     #显示设置当前工作目录为项目的根目录




class DataProcessor:
    def __init__(self,dataset_name,dataset_cache_dir,model_name,tokenizer_cache_dir,max_len=256,batch_size=32):
        
        self.dataset_name=dataset_name      #数据集在HuggingFace中的名称
        self.dataset_cache_dir=dataset_cache_dir        #数据集下载位置
        self.model_name=model_name      #分词器所属模型在HuggingFace中的名称
        self.tokenizer_cache_dir=tokenizer_cache_dir        #模型分词器下载位置
        self.max_len=max_len        #最大长度(后续编码对于超过的长度直接截断,对于不足的长度进行填补)
        self.batch_size=batch_size      #同一个批次处理的样本数量
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,cache_dir=tokenizer_cache_dir)     #加载模型分词器
        
        print(self.tokenizer("hello python"))      
        #可以通过print查看tokenizer基本的输出格式
        #可以观察到返回一个字典,字典的三个键分别为"input_ids"(输入数据)、"token_type_ids"(反映两个语句之间的关系)、"attention_mask"(反映语句是否进行过填充)
        
    def load_data(self):
        
        dataset=load_dataset(self.dataset_name,cache_dir=self.dataset_cache_dir)        #加载数据集
        
        print(dataset)      
        #可以通过print查看HuggingFace中dataset的基本结构
        #可以发现结构包括"train"、"test"、"unsupervised"三种数据集,每种数据集还分成"text"与"label"两种类型
        
        print(dataset["train"][0])      
        #可以通过print查看训练集或测试集单一样本的基本结构
        #可以发现结构为一个字典,包括"text"与"label"两个键,text对应一个文本字符串,label对应0或1
            
        return dataset
    
    
    def encode_data(self,dataset):
        
        train_input_ids=[]
        train_attention_mask=[]
        train_label=[]
        
        test_input_ids=[]
        test_attention_mask=[]
        test_label=[]
        
        for example in dataset["train"]:
            
            encoding=self.tokenizer(example["text"],padding="max_length",truncation=True,max_length=self.max_len)
            train_input_ids.append(encoding["input_ids"])
            train_attention_mask.append(encoding["attention_mask"])
            train_label.append(example["label"])
        
        train_encoded_dict={"input_ids":train_input_ids,"attention_mask":train_attention_mask,"label":train_label}    
            
        for example in dataset["test"]:
            
            encoding=self.tokenizer(example["text"],padding="max_length",truncation=True,max_length=self.max_len)
            test_input_ids.append(encoding["input_ids"])
            test_attention_mask.append(encoding["attention_mask"])
            test_label.append(example["label"])
        
        test_encoded_dict={"input_ids":test_input_ids,"attention_mask":test_attention_mask,"label":test_label}    
             
        return train_encoded_dict,test_encoded_dict 
            
            
        """
        问题记录:最初我想使用map不断调用tokenize_function对整个文本进行编码,但是最终返回的是list类型,而非字典类型,同时未能成功对文本进行编码,目前不清楚问题在哪里
        原本的代码:
        #tokenized_train=dataset["train"].map(tokenize_function,batched=True,cache_file_name="./imdb_data/tokenized_train.arrow")
        #tokenized_test=dataset["test"].map(tokenize_function,batched=True,cache_file_name="./imdb_data/tokenized_test.arrow")
        
        #batched=True表示tokenize_function会一次性处理整个批次,而不是逐个样本进行处理
        #map会根据数据集的大小自动将数据拆分为多个批次(并非个人指定的self.batch_size),然后将每个批次的数据传递给tokenize_function
        #属于HuggingFace标准写法
        #值得注意的是在使用map,filter等函数时,HuggingFace的dataset库会自动以哈希名字缓存中间转换结果,因此每次统一文件名字可以防止每次编译生成重复文件
        return tokenized_train,tokenized_test
        """
    
    def create_dataloader(self,train_encoded_dict,test_encoded_dict):
        
        print(torch.tensor(train_encoded_dict["input_ids"]).shape)      #可以通过print查看input_ids对应的数据形状,正常应该为torch.Size([25000,max_len]),是二维张量
        print(torch.tensor(train_encoded_dict["attention_mask"]).shape)     #可以通过print查看attention_mask对应的数据形状,正常应该为torch.Size([25000,max_len]),是二维张量
        print(torch.tensor(train_encoded_dict["label"]).shape)      #可以通过print查看label对应的数据形状,正常应该为torch.Size([25000]),是一维张量
    
        train_dataset=TensorDataset(
                                    torch.tensor(train_encoded_dict["input_ids"]),
                                    torch.tensor(train_encoded_dict["attention_mask"]),
                                    torch.tensor(train_encoded_dict["label"])
                                   )
        #TensorDataset的功能是将多个张量组合成一个数据集,其格式是包含多个数据的容器
        #此处TensorDataset将返回三元组,分别是"input_ids"、"attention_mask"、"label"对应的二维或一维张量
        #此处对训练集进行操作
        
        test_dataset=TensorDataset(
                                   torch.tensor(test_encoded_dict["input_ids"]),
                                   torch.tensor(test_encoded_dict["attention_mask"]),
                                   torch.tensor(test_encoded_dict["label"])
                                  )
        #此处对测试集进行操作
        
        train_dataloader=DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=4)
        #TensorDataset将数据整合为三元组,DataLoader会根据batch_size将三元组每一个元素对应的张量进行划分
        #具体而言,每个batch中"input_ids"对应的张量被分割为(batch_size,max_len)形状,每个batch中"attention_mask"对应的张量被分割为(batch_size,max_len)形状,每个batch中"label"对应的张量被分割为(batch_size,)
        #此处包装训练集(DataLoader包装之后依然是三元组结构)
        test_dataloader=DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=4)
        #此处包装测试集(DataLoader包装之后依然是三元组结构)
        
        return train_dataloader, test_dataloader
    
    
    
    
class BertClassifier(nn.Module):
    def __init__(self,model_name,model_cache_dir,hidden_size=768,classes_num=2,dropout=0.1):        #Bert模型hidden_size一般默认768
        
        super().__init__()      #调用父类并进行初始化操作
        
        self.bert=AutoModel.from_pretrained(model_name,cache_dir=model_cache_dir)      #加载Bert原始模型
        self.dropout=nn.Dropout(dropout)        #创建一个dropout层,防止过拟合
        self.linear=nn.Linear(hidden_size,classes_num)
        
        
    def forward(self,input_ids,attention_mask):
        
        output=self.bert(input_ids=input_ids,attention_mask=attention_mask)
        #input_ids是将输入的文本按照tokenizer进行编号替换之后的输入
        #attention_mask能标识哪些token是原文本信息,哪些是为了长度对齐填充生成的 
        cls_output=output.last_hidden_state[:,0,:]      
        #last_hidden只获取最后一个隐藏层的输出
        #output.last_hidden_state的输出的形状为(batch_size,seq_length,hidden_size)  
        #bert对于每次输入的文本,除了会划分为多个token以外,还会在整个序列最前面添加cls标签,在最后添加esp标签,cls总结关于情感分类的信息,esp总结语句之间的联系
        #对于这个情感分类问题,只需关注cls的输出结果就行了,因此cls_output对应(batch_size,1,hidden_size)
        cls_output=self.dropout(cls_output)    
        logits=self.linear(cls_output)      
        
        return logits




class Trainer():
    def __init__(self,model,train_loader,test_loader,criterion,optimizer,scheduler,epochs_num):        
        
        self.model=model
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.criterion=criterion
        self.optimizer=optimizer
        self.scheduler=scheduler        #scheduler是学习率调度器,可以动态调整学习率
        self.epochs_num=epochs_num        #epochs_num表示
    
    
    def one_epoch_train(self):
        
        self.model.train()      #将模型设置为训练模式
        
        total_loss=0        #total_loss是记录当前epoch中所有批次的损失之和
        
        for batch_idx,batch in enumerate(tqdm(self.train_loader,desc="Training")):      
        #batch_ids是每一个批次的编号,而batch则是当前批次的数据(每个batch包含batch_size个数据)
        #此处batch结构可以表示为{"input_ids":tensor[……],"attention_mask":tensor[……],"label":tensor[……]}
        #tqdm函数用于显示加载数据的实时进度,desc则是自定义的标语
            
            print(batch)        
            #可以通过print查看每一个批次的基本结构
            #正常输出应该是三个tensor,分别对应"input_ids","attention_mask","label"
            
            input_ids=batch[0]      #"input_ids"是三元组的第0个元素
            attention_mask=batch[1]     #"attention_mask"是三元组的第1个元素
            label=batch[2]      #"label"是三元组的第2个元素
            
            self.optimizer.zero_grad()      #每次反向传播之前清除上一个batch的梯度
            logits=self.model.forward(input_ids,attention_mask)     #前向传播获得一个二维向量
            loss=self.criterion(logits,label)       #计算损失
            loss.backward()     #根据当前batch的损失计算梯度
            self.optimizer.step()       #根据反向传播计算的梯度更新参数
            #思考:loss.backward()计算梯度的前提是知道模型内部结构,但是代码没有显示相关信息传递,loss怎么得知model的内部结构呢?
            #     同样的问题,loss.backward()计算的梯度没有在代码中传递给optimizer,optimizer是怎么得知梯度的数据并进行参数更新呢?
            #解答:loss.backward()会根据loss的计算过程,自动追踪和分析模型中所有参与运算的参数,并计算出它们的梯度(依赖前向传播中构建的计算图)
            #     pytorch巧妙地构建了"计算图",也就是说,不论前向传播还是反向传播的各种库函数,都在执行的过程中获取到"计算图"信息,并不断更改信息(构建"计算图"),由此实现信息共享,类比于这些函数公用同一个内存
            
            total_loss+=loss.item()
            
            print(f"Batch_idx:{batch_idx}/{len(self.train_loader)}---Loss:{loss.item():.4f}")       #输出当前批次batch_idx以及损失
        
        return total_loss/len(self.train_loader)      #返回平均损失


    def one_epoch_evaluate(self):
        
        self.model.eval()       #将模型设置为评估模式
        
        total_loss=0        #累计所有批次的损失,用于计算平均损失
        total=0     #累计所有样本的数量
        correct=0       #统计分类正确的次数
        
        with torch.no_grad():       #禁用梯度计算(评估阶段只需要正向传播,不需要计算梯度)
            for batch in tqdm(self.test_loader,desc="Evaluating"):
                
                input_ids=batch[0]
                attention_mask=batch[1]
                label=batch[2]
                
                logits=self.model.forward(input_ids,attention_mask)
                loss=self.criterion(logits,label)
                total_loss+=loss.item()
                preds=torch.argmax(logits,dim=1)        
                #logits形状为(batch_size,classes_num)
                #dim=0表示沿着批次的维度(即每列最大值)进行操作,dim=1表示沿着每一个样本的类别维度(即每行最大值)进行操作
                correct+=(preds==label).sum().item()        #统计每一个批次的分类正确次数
                total+=label.size(0)        #统计当前批次的数据数量
        
        average_loss=total_loss/len(self.test_loader)     #计算所有数据的平均损失
        accuracy=correct/total      #计算所有数据的正确率
        
        return average_loss,accuracy
    
    
    def all_epochs_train(self):
        for epoch in range(self.epochs_num):
            
            print(f"Epoch:{epoch}/{self.epochs_num}")
            train_loss=self.one_epoch_train()
            average_loss,accuracy=self.one_epoch_evaluate()
            print(f"Train-Loss:{train_loss}     |   test-Loss:{average_loss}    |   Test-Accuracy:{accuracy}")
            
            self.scheduler.step()       #在每一个epoch结束时更新学习率(通过逐渐减小学习率可以使得模型后续学习更加精细)
    
    
    def save_model(self,path):
        
        torch.save(self.model.state_dict(),path)        #state_dict()表示保存模型参数
       
       
        
    
dataprocessor=DataProcessor("stanfordnlp/imdb","./imdb_data","bert-base-uncased","./hf-tokenizer")
dataset=dataprocessor.load_data()
tokenized_train,tokenized_test=dataprocessor.encode_data(dataset)
train_loader,test_loader=dataprocessor.create_dataloader(tokenized_train,tokenized_test)

model=BertClassifier("bert-base-uncased","./hf-model",hidden_size=768,classes_num=2,dropout=0.1)

criterion=nn.CrossEntropyLoss()
optimizer=AdamW(model.parameters(),lr=2e-5)
scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)
#step_size表示学习率更新频率(此处表示每一个epoch更新一次)
#gamma表示每次更新学习率时缩小为原来的10%
trainer=Trainer(model,train_loader,test_loader,criterion,optimizer,scheduler,epochs_num=3)
trainer.all_epochs_train()
trainer.save_model("./trained-model/trained_parameters.pth")        
#保存模型
#注意:此处保存模型需要明确路径和文件名,而非文件夹,文件后缀通常为.pth或者.pt