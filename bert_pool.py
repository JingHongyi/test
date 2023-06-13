from transformers import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
texts = [["这是一个实验文本，看看如何进行分析的。",
         "我们的爱可以重来，我们的爱值得等待。",
         "你说你想要逃，偏偏注定要落脚。"],
         ["这是一个实验文本，看看如何进行分析的。",
         "我们的爱可以重来，我们的爱值得等待。",
         "你说你想要逃，偏偏注定要落脚。"]
]
encodes = []
for i in range(len(texts)):
    text = texts[i]
    res = []
    for j,sen in enumerate(text):
        tmp = tokenizer(sen,padding='max_length',max_length=20,truncation=True)['input_ids']
        res.append(tmp)
    encodes.append(res)
encodes = torch.Tensor(encodes)
a = torch.nn.MaxPool2d((3,1))
print(a(encodes.float()).squeeze(1))
