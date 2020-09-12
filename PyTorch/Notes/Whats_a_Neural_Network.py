# -*- coding: utf-8 -*-
"""
Shallow Neural Networks
Neural Networks in one dimension

2 layer model
"""


import torch
import torch.nn as nn

### Neural Network ###

class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(D_in,H)
        self.linear2 = nn.Linear(H,D_out)
        
    def forward(self,x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x 
    
# Nural netowrk object

model_Mod = Net(1,2,1)    
x = torch.tensor([0.0])
y_hat = model_Mod(x)

x = torch.tensor([[0.0], [2.0], [3.0]])
y_hat = model_Mod(x)
y_hat = y_hat>0.5 # Apply threshold to get discrete value

model_Mod.state_dict() # This method has the model parameters.


# Neural Network using nn.Sequential

model_seq = torch.nn.Sequential(nn.Linear(1,2), torch.nn.Sigmoid(), nn.Linear(2,1), torch.nn.Sigmoid())



### Training the model ###


# Create the data

X = torch.arange(-20,20,1).view(-1,1).type(torch.FloatTensor)
Y = torch.zeros(X.shape[0])
Y[(X[:,0]>-4) & (X[:,0]<4)] = 1.0

# Create training function

def train(Y,X,model,optimizer,criterion,epochs=5):
    cost = []
    total = 0
    for epoch in range(epochs):
        total = 0
        for y,x in zip(Y,X):
            yhat = model(x)
            loss = criterion(yhat,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #cumulative loss
            total += loss.item()
        cost.append(total)
    return cost

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model_Mod.parameters(),lr=0.01)
cost = train(Y,X,model_Mod,criterion,epochs=5)

