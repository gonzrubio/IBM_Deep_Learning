# -*- coding: utf-8 -*-
"""
Shallow Neural Networks /
2.2 Neural Networks More Hidden Neurons
Quiz: More Hidden Neurons
"""

import torch
import torch.nn as nn

# 1) Consider the following neural network model or class
# How many hidden neurons does the following neural network object have?

class Net(nn.Module):

    def __init__(self,D_in,H,D_out):
    
        super(Net,self).__init__()
        
        self.linear1=nn.Linear(D_in,H)
        
        self.linear2=nn.Linear(H,D_out)

    def forward(self,x):
    
        x=torch.sigmoid(self.linear1(x))
        
        x=torch.sigmoid(self.linear2(x))
        
        return x


model=Net(1,6,1)
print(list(model.named_parameters()))