# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from . import Linear


class FullFusion(nn.Module):
    def __init__(self,input_size,dropout=0.2,hidden_size=200):
        super().__init__()
        self.dropout = dropout
        self.fusion1 = Linear(input_size * 2, hidden_size*2, activations=True)
        self.fusion2 = Linear(input_size * 2, hidden_size*2, activations=True)
        self.fusion3 = Linear(input_size * 2, hidden_size*2, activations=True)
        self.fusion = Linear(hidden_size * 6, hidden_size*2, activations=True)

    def forward(self, x, align):
        x1 = f.softmax(self.fusion1(torch.cat([x, align], dim=-1)), dim = -1)
        x2 = f.softmax(self.fusion2(torch.cat([x, x - align], dim=-1)), dim = -1)
        x3 = f.softmax(self.fusion3(torch.cat([x, x * align], dim=-1)), dim = -1)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = f.dropout(x, self.dropout, self.training)
        return self.fusion(x)
