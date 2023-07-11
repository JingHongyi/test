import torch.nn as nn
import torch
from transformers import (
                    BertTokenizer,
                    BertModel,
                    AutoTokenizer,
                    AutoModel
                    )

texts = [
    '相信吧，快乐的日子将会来临！',
    '心儿永远向往着未来，现在却常是忧郁。',
    '一切都是瞬息，一切都将过去，而那过去了的，就会成为亲切的怀念。',
    '千里冰封，万里雪飘，望长城内外，惟余莽莽。',
    '大河上下，顿失滔滔，山舞银蛇，原驰蜡象，欲与天公试比高。',
    '须晴日，看红装素裹，分外妖娆。'
         ]
pretrained = 'bert_pretrain/roberta-chinese'
tokenizer = AutoTokenizer.from_pretrained(pretrained)
input_ids = []
input_ids2 = []
mask = []
for i in range(len(texts)):
    token = tokenizer(texts[i],max_length=32,padding='max_length',truncation=True)
    input_ids.append(token['input_ids'])
    print(token['input_ids'])
    mask.append(token['attention_mask'])
input_ids = torch.Tensor(input_ids).long()
mask = torch.Tensor(mask)
model = AutoModel.from_pretrained(pretrained)
output = model(input_ids,mask)
print(output.last_hidden_state.shape)
output = output.last_hidden_state.reshape(-1,3,32,768)
print(output.shape)
# conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(8,768))
# output = conv1(output).squeeze(3)
# print(output.shape)
# size = output.shape[2]
# max_pool1d = nn.MaxPool1d(kernel_size=size)
# output = max_pool1d(output)
# print(output.shape)
