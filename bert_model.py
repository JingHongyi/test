from transformers import BertTokenizer,BertModel
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import json
from tqdm import tqdm
import os
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# dataset with 2 sentence
class Mydata(Dataset):
    def __init__(self,texts):
        super(Mydata).__init__()
        self.data = []
        self.mask = []
        self.label = []
        for i,content in enumerate(texts):
            self.label.append(int(content[0]))
            if len(content[1])>=80:
                res_ids, res_mask = [], []
                token1 = tokenizer(content[1][:80],padding='max_length',max_length=80,truncation=True)
                res_ids.append(token1['input_ids'])
                res_mask.append(token1['attention_mask'])
                token2 = tokenizer(content[1][-80:],padding='max_length',max_length=80,truncation=True)
                res_ids.append(token2['input_ids'])
                res_mask.append(token2['attention_mask'])
                self.data.append(res_ids)
                self.mask.append(res_mask)
            else:
                res = []
                mask = []
                token = tokenizer(content[1],padding='max_length',max_length=80,truncation=True)
                res.append(token['input_ids'])
                res.append(token['input_ids'])
                mask.append(token['attention_mask'])
                mask.append(token['attention_mask'])
                self.data.append(res)
                self.mask.append(mask)
        self.label = torch.Tensor(self.label)
        self.data = torch.Tensor(self.data)
        self.mask = torch.Tensor(self.mask)
    
    def __getitem__(self, idx):
        input_ids,attention_mask,label = self.data[idx],self.mask[idx],self.label[idx]
        return input_ids,attention_mask,label
    
    def __len__(self):
        return len(self.data)

## dataset with one sentence(512 token)
# class Mydata(Dataset):
#     def __init__(self,texts):
#         super(Mydata).__init__()
#         self.data = []
#         self.label = []
#         for i,content in enumerate(texts):
#             self.label.append(int(content[0]))
#             res = tokenizer(content[1],padding='max_length',max_length=512,truncation=True,return_tensors='pt')
#             self.data.append(res)
    
#     def __getitem__(self, idx):
#         x, y = self.data[idx], self.label[idx]
#         return x, y
    
#     def __len__(self):
#         return len(self.data)

class BertClassifier(nn.Module):
    def __init__(self,dropout=0.5) -> None:
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.mlp = nn.Linear(768,2)
        self.max_pool = nn.MaxPool2d((2,1))
        self.dp = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self,input_ids,atten_mask):
        _, pool_out = self.bert(input_ids,atten_mask,return_dict=False)
        pool_out = pool_out.reshape(-1,2,768)
        pool_out = self.max_pool(pool_out).reshape(-1,768)
        out = self.soft(self.relu(self.mlp(self.dp(pool_out))))
        return out

one_texts = []
zero_texts = []
with open('review.csv','r') as f:
    for line in f.readlines():
        line = line.strip().split(',')
        if line[0]!='label':
            if line[0]=='0':
                zero_texts.append(line)
            else:
                one_texts.append(line)

train_text = one_texts[:500] + zero_texts[:500]
test_text = one_texts[-10:] + zero_texts[-10:]
train_data = Mydata(train_text)
test_data = Mydata(test_text)

train_dataloader = DataLoader(train_data,batch_size=32,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=8,shuffle=True)

model = BertClassifier()
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(),lr=1e-5)

for epo in range(100):
    loss_list = []
    for i,(input_ids,attention_mask,label) in tqdm(enumerate(train_dataloader)):
        ids_shape = input_ids.shape
        atten_shape = attention_mask.shape
        input_ids = input_ids.reshape(-1,ids_shape[-1]).long()
        attention_mask = attention_mask.reshape(-1,atten_shape[-1])
        opt.zero_grad()
        # input_ids = data['input_ids'].squeeze(1)
        # attention_mask = data['attention_mask']
        output = model(input_ids,attention_mask)
        loss = loss_fn(output,label.long())
        loss.backward()
        opt.step()
        loss_list.append(loss.item())
    loss_list = np.array(loss_list)
    print("Epoch {} loss {}".format(epo+1,np.mean(loss_list)))
    with torch.no_grad():
        total = 0
        right = 0
        loss_list = []
        for i,(input_ids,attention_mask,label) in tqdm(enumerate(test_dataloader)):
            ids_shape = input_ids.shape
            atten_shape = attention_mask.shape
            input_ids = input_ids.reshape(-1,ids_shape[-1]).long()
            attention_mask = attention_mask.reshape(-1,atten_shape[-1])
            output = model(input_ids,attention_mask)
            loss = loss_fn(output,label.long())
            loss_list.append(loss.item())
            total += len(label)
            right += (torch.argmax(output,dim=1)==label).sum()
        loss_list = np.array(loss_list)
        print('Eval loss {}'.format(np.mean(loss_list)))
        print('Accuracy {}'.format(right*1.0/total))
        
            

    

    
        