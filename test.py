import numpy as np
import torch


b = torch.tensor([[[1,1,1],[1,1,1]]])
a = torch.tensor([[[2,0,2],[2,2,0]]])
print(a.shape)
print(b.shape)


print(a[b].shape)

