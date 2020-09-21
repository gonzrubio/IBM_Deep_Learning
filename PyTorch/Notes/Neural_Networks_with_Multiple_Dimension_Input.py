# -*- coding: utf-8 -*-
"""
Shallow Neural Networks
2.3 Neural Networks with Multiple Dimension Input
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class XOR_Data(Dataset):
    def __init__(self, N_s=100):
        self.x = torch.zeros((N_s, 2))
        self.y = torch.zeros((N_s, 1))
        for i in range(N_s // 4):
            self.x[i,:] = torch.tensor([0.0, 0.0])
            self.y[i,0] = torch.tensor([1.0])

            self.x[i + N_s // 4,:] = torch.tensor([0.0, 1.0])
            self.y[i + N_s // 4,0] = torch.tensor([1.0])            

            self.x[i + N_s // 2,:] = torch.tensor([1.0, 0.0])
            self.y[i + N_s // 2,0] = torch.tensor([1.0])              

            self.x[i + 3 * N_s // 4,:] = torch.tensor([1.0, 1.0])
            self.y[i + 3 * N_s // 4,0] = torch.tensor([1.0]) 
            
            self.x = self.x + 0.01 * torch.randn((N_s, 2))
        self.len = N_s
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
class Net(nn.Module):
    def __init__(self,D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        
    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


def train(data_set,model,criterion,train_loader,optimizer,epochs=5):
    cost = []
    acc = []
    for epoch in range(epochs):
        total = 0
        for y,x in train_loader:
            loss = criterion(model(x),y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        acc.append(accuracy(model,data_set))
        cost.append(total)
    return cost   


model = Net(2,4,1) # 2 input dimensions and 4 neurons in the hidden layer
criterion = nn.BCELoss()        
dataset = XOR_Data()
train_loader = DataLoader(dataset=dataset, batch_size=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
cost = train(dataset,model,criterion,train_loader,optimizer)



    