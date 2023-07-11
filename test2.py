import torch
import random
a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
seed = 5
random.seed(seed)
random.shuffle(a)
random.seed(seed)
random.shuffle(b)
print(a)
print(b)