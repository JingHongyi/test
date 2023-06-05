import numpy as np
from transformers import BertTokenizer
text1 = "我们的爱可以重来，[SEP]我们的爱值得等待。"
text2 = "我们的爱可以重来，我们的爱值得等待。"
text3 = "[CLS]"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(tokenizer(text1)['input_ids'])
print(tokenizer(text2)['input_ids'])