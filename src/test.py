import torch
import torch.nn as nn

a = torch.randn(3,10)
maxpool = nn.MaxPool1d(2,stride=2)
c = maxpool(a)
d = torch.randn(10)
print(a)
print(c)
maxpool(d)