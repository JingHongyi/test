import torch
import random
import os
from transformers import BertTokenizer,LongformerModel,BertModel,LongformerTokenizer,AutoTokenizer
tokenizer = BertTokenizer.from_pretrained('bert_pretrain/longformer-chinese')
a = '为什么，越珍贵，越浪费。致命的伤，诞生于亲密。'
output = tokenizer(a,max_length=20000,padding='max_length',truncation=True)
model1 = BertModel.from_pretrained('bert_pretrain/roberta-chinese')
model2 = LongformerModel.from_pretrained('bert_pretrain/longformer-chinese')
# print(model1(torch.Tensor(output['input_ids']).long().unsqueeze(0)).pooler_output)
print(model2(torch.Tensor(output['input_ids']).long().unsqueeze(0)).pooler_output.shape)