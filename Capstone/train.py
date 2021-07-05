"""
Created on Sat Jul  3 11:04:52 2021

@author: gonzr
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils import DenominationsData

device = "cuda:0" if torch.cuda.is_available() else "cpu"

##############################################################################
#                                                                            #
#                                    Data                                    #
#                                                                            #
##############################################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])

train_dir = 'data/training/'
train_csv_file = 'data/training/training_labels.csv'
train_dataset = DenominationsData(train_csv_file, train_dir, transform=composed)

val_dir = 'data/training/'
val_csv_file = 'data/validation/validation_labels.csv'
validation_dataset = DenominationsData(val_csv_file, val_dir, transform=composed)


##############################################################################
#                                                                            #
#                                   Model                                    #
#                                                                            #
##############################################################################

model = resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Replace the output layer model.fc of the neural network with
# a nn.Linear object, to classify 7 different bills. For the parameters
# in_features  remember the last hidden layer has 512 neurons.

fc = torch.empty(size=(512, 7))
fc = nn.init.kaiming_uniform_(fc, mode='fan_in', nonlinearity='relu')
model.fc = nn.Parameter(fc)
# model.fc = nn.Linear(512, 7)


##############################################################################
#                                                                            #
#                                Hyperparameters                             #
#                                                                            #
##############################################################################

criterion = nn.CrossEntropyLoss()

training_loader = DataLoader(train_dataset, batch_size=15, shuffle=True,
                             num_workers=2, pin_memory=True)
training_loader = DataLoader(train_dataset, batch_size=10, shuffle=False,
                             num_workers=2, pin_memory=True)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

epochs = 20


##############################################################################
#                                                                            #
#                                  Training                                  #
#                                                                            #
##############################################################################

correct = 0
loss = []       # Per epoch
accuracy = []   # " "

for epoch in range(epochs):

    loss_batch = 0.0
    for batch, (x, y) in enumerate(training_loader):
        x = x.to(device)
        y = y.to(device)
        loss_batch += criterion(model(x), y).item()

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        scheduler.step()

    loss.append
























