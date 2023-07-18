import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import os
from transformers import BertTokenizer,LongformerTokenizer
from dataset.news_data import News
from dataset.news_long_data import NewsLong
from model.roberta import BertClassifier
from model.roberta_cnn import BertCNNClassifier
from model.longformer import LongformerClassifier
import numpy as np

train_dataset = NewsLong(train=True)
test_dataset = NewsLong(train=False)

train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True)

# model = BertClassifier()
model = LongformerClassifier(num_class=5,dropout=0.5)
optim = torch.optim.Adam(model.parameters(),lr=0.00001)
loss_fn = nn.CrossEntropyLoss()
model.train()

for epo in range(50):
    loss_list = []
    print('Epoch {}'.format(epo+1))
    for i,(input_ids,attention_mask,labels) in enumerate(train_dataloader):
        input_ids_shape = input_ids.shape
        input_ids = input_ids.reshape(-1,input_ids_shape[-1])
        attention_mask_shape = attention_mask.shape
        attention_mask = attention_mask.reshape(-1,attention_mask_shape[-1])
        output = model(input_ids,attention_mask)
        optim.zero_grad()
        loss = loss_fn(output,labels)
        loss_list.append(loss.item())
        loss.backward()
        optim.step()
        print("batch {} loss {}".format(i+1,loss.item()))
    loss_list = np.array(loss_list)
    print("Epoch Loss {}".format(np.mean(loss_list)))
    with torch.no_grad():
        total = 0
        right = 0
        loss_fn = []
        for i,(input_ids,attention_mask,labels) in enumerate(test_dataloader):
            input_ids_shape = input_ids.shape
            input_ids = input_ids.reshape(-1,input_ids_shape[-1])
            attention_mask_shape = attention_mask.shape
            attention_mask = attention_mask.reshape(-1,attention_mask_shape[-1])
            output = model(input_ids,attention_mask)
            loss_list.append(loss_fn(output,labels).item())
            total += input_ids.shape[0]
            right += (torch.argmax(output)==labels).sum()
        loss_list = np.array(loss_list)
        print("Test Loss {}".format(np.mean(loss_list)))
        print("Test Accuracy {}".format(right*1.0/total))
            