import torch
import cProfile
import pstats
from transformers import BertModel
a = torch.Tensor([[[1,2,3],
                   [2,3,4]],
                   [[4,5,6],
                    [6,7,8]]])
a = a.reshape(-1,3)
a = a.reshape(-1,2,3)
b = torch.nn.MaxPool2d((2,1))
print(b(a))