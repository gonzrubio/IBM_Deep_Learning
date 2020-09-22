# -*- coding: utf-8 -*-
"""
Shallow Neural Networks
Neural Networks more hidden Neurons

One hidden layer with six neurons
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Class to get our dataset

class Data(Dataset):
    def __init__(self):
        self.x = torch.linspace(-20,20,100).view(-1,1)
        self.len = self.x.shape[0]
        self.y = torch.zeros(self.x.shape[0]).view(-1,1)
        self.y[(torch.abs(self.x[:,0])>5) & (torch.abs(self.x[:,0])<10)] = 1
    
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len 
    
    
# Class for creating the model    
        
class Net(nn.Module):
    def __init__(self,D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x

# Function for training the model

def train(model,criterion,train_loader,optimizer,epochs=5):
    cost = []
    for epoch in range(epochs):
        total = 0
        for batch_idx, (y,x) in enumerate(train_loader):
            print('epoch {}, batch_idx {} , batch len {}'.format(epoch, batch_idx, len(y)))
            loss = criterion(model(x),y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        cost.append(total)
    return cost        
        
        
# Training

model = Net(1,6,1)
criterion = nn.BCELoss()        
dataset = Data()
train_loader = DataLoader(dataset=dataset, batch_size=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
cost = train(model,criterion,train_loader,optimizer)



        
        