import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from transformers import BertTokenizer,BertModel

class BertClassifier(nn.Module):
    def __init__(self,dropout=0.5):
        super(BertClassifier,self).__init__()
        self.bert = BertModel.from_pretrained('./bert_pretrain/roberta-chinese')
        self.mlp = nn.Linear(768,3)
        self.max_pool = nn.MaxPool2d((3,1))
        self.dp = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self,input_ids,atten_mask):
        _, pool_out = self.bert(input_ids,atten_mask,return_dict=False)
        pool_out = pool_out.reshape(-1,3,768)
        pool_out = self.max_pool(pool_out).reshape(-1,768)
        out = self.soft(self.relu(self.mlp(self.dp(pool_out))))
        return out
