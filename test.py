import torch
import cProfile
import pstats
a = torch.Tensor([[1,1,5,5],
                  [5,6,7,10],
                  [2,2,3,5]])

b = torch.Tensor([[2,2,4,4],
                  [2,2,4,4],
                  [1,6,4,7]])
center = (b[:,:2] + b[:,2:]) / 2
print(center[:,None,0] >= a[:,0])
# res = (center[:,None,0] >= a[:,0]) & (center[:,None,0] <= a[:,2]) & (center[:,None,1] >= a[:,1]) & (center[:,None,1] <= a[:,3])
# print(res.any(dim=1))