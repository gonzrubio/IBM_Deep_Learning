# -*- coding: utf-8 -*-
"""
Shallow Neural Networks /
2.3 Neural Networks with Multiple Dimension Inputs
Quiz: Neural Networks with Multiple Dimension Inputs
"""

import torch
import torch.nn as nn

# 1) How many dimensions is the input for the following neural network object?

class Net(nn.Module):

    def __init__(self,D_in,H,D_out):    
        super(Net,self).__init__()        
        self.linear1=nn.Linear(D_in,H)    
        self.linear2=nn.Linear(H,D_out)

    def forward(self,x):   
        x=torch.sigmoid(self.linear1(x))       
        x=torch.sigmoid(self.linear2(x))       
        return x

model = Net(4,10,1)
print(model)


# 2) How many dimensions is the input for the following neural network object?

class Net(nn.Module):

    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)

    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        return x

model = Net(3,4,1)
print(model)


