"""
Created on Sat Jul  3 11:04:52 2021

@author: gonzr
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils import DenominationsData

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

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

val_dir = 'data/validation/'
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
model.fc = nn.Linear(512, 7)
model.to(device)

##############################################################################
#                                                                            #
#                                Hyperparameters                             #
#                                                                            #
##############################################################################

train_bsize = 10
val_bsize = 15
criterion = nn.CrossEntropyLoss()

training_loader = DataLoader(train_dataset, batch_size=val_bsize, shuffle=True,
                             pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=train_bsize,
                               shuffle=False, pin_memory=True)

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=7, eta_min=0)
epochs = 20


##############################################################################
#                                                                            #
#                                  Training                                  #
#                                                                            #
##############################################################################

loss = []       # Per epoch
accuracy = []   # " "

for epoch in range(epochs):

    # Training
    loss_epoch = 0.0
    for batch, (x, y) in enumerate(training_loader):
        x = x.to(device)
        y = y.to(device)
        # model.to(device)
        loss_batch = criterion(model(x), y)
        loss_epoch += loss_batch.item()

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        scheduler.step()
        print(f"{epoch}.{batch} {loss_batch.item()}")
    loss.append(loss_epoch / len(training_loader))

    # Validation
    with torch.no_grad():
        model.eval()
        correct = 0
        for batch, (x, y) in enumerate(validation_loader):
            x = x.to(device)
            y = y.to(device)
            # model.to(device)
            y_hat = model(x).max(dim=1)[1]
            correct += (y_hat == y).sum().item()
        accuracy.append(correct / len(validation_dataset))
        model.train()
    print(f"Epoch: {epoch} accuracy: {accuracy[-1]}")


plt.plot(loss)
plt.ylabel("Average loss")
plt.xlabel("epoch")
plt.show()

plt.plot(accuracy)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()
