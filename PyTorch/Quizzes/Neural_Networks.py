# -*- coding: utf-8 -*-
"""
Shallow Neural Networks /
2.1 Neural Networks in One Dimension
Quiz: Neural Networks
"""

import torch
import torch.nn as nn

# 1) How many hidden neurons does the following neural  network object have?

class Net(nn.Module):

    def __init__(self,D_in,H,D_out):
    
        super(Net,self).__init__()
        
        self.linear1=nn.Linear(D_in,H)
        
        self.linear2=nn.Linear(H,D_out)
    
    def forward(self,x):
    
        x=torch.sigmoid(self.linear1(x))
        
        x=torch.sigmoid(self.linear2(x))
    
        return x

model = Net(1,3,1)
print(list(model.named_parameters()))


# 2) How many hidden neurons does the following neural network object have?

class Net(nn.Module):

    def __init__(self,D_in,H,D_out):
    
        super(Net,self).__init__()
        
        self.linear1=nn.Linear(D_in,H)
        
        self.linear2=nn.Linear(H,D_out)
    
    def forward(self,x):
    
        x=torch.sigmoid(self.linear1(x))
        
        x=torch.sigmoid(self.linear2(x))
        
        return x

model = Net(1,6,1)
print(list(model.named_parameters()))


# 3) Whats wrong with the following function?

class Net(nn.Module):

    def __init__(self,D_in,H,D_out):

        super(Net,self).__init__()
        
        self.linear1=nn.Linear(D_in,H)
        
        self.linear2=nn.Linear(H,D_out)

    def forward(self,x):
        
        #x=torch.sigmoid(linear1(x)) wrong
        x=torch.sigmoid(self.linear1(x))
        
        #x=torch.sigmoid(linear2(x)) wrong
        x=torch.sigmoid(self.linear2(x))
        
        return x

# 4) What does the folliwng line of code do?

# torch.sigmoid(self.linear1(x)) - Applies sigmoid activation to the linear transformation.