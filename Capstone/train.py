"""
Created on Sat Jul  3 11:04:52 2021

@author: gonzr
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random

from torchvision import transforms
from torchvision.models import resnet18
from torchvision.models import densenet121
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from utils import DenominationsData

device = "cuda:0" if torch.cuda.is_available() else "cpu"
random.seed(0)

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


##############################################################################
#                                                                            #
#                                  Plots                                     #
#                                                                            #
##############################################################################

# Run the function to plot image, print the predicted label and
# print a string indicate whether it has been correctly classified
# or mis-classified.

look_up = {0: 'predicted: $5', 1: 'predicted: $10', 2: 'predicted: $20',
           3: 'predicted: $50', 4: 'predicted: $100', 5: 'predicted $200',
           6: 'predicted $500'}


def plot_random_image():
    """Plot 5 random images from the validation set."""
    model.eval()
    with torch.no_grad():
        for number in random.sample(range(70), 5):
            X, Y = validation_dataset[number]
            y_hat = model(X[None, :, :, :].to(device)).max(dim=1)[1]
            title = "Correctly" if y_hat.item() == Y else "Incorrectly"
            plt.imshow(X.cpu().permute(1, 2, 0), vmin=0, vmax=255)
            plt.title(f"Predicted label: {y_hat.item()}, {title} classified")
            plt.show()
            print(f"{look_up[y_hat.item()]}, {title} classified")
    model.train()


plot_random_image()


##############################################################################
#                                                                            #
#                                Densenet121                                 #
#                                                                            #
##############################################################################

model_des = densenet121(pretrained=True)

for param in model_des.parameters():
    param.requires_grad = False

# Replace the output layer model_des.fc of the neural network with
# a nn.Linear object, to classify 7 different bills. For the parameters
# in_features  remember the last hidden layer has 1024 neurons.
model_des.classifier = nn.Linear(1024, 7)
model_des.to(device)

train_bsize = 15
val_bsize = 10
criterion = nn.CrossEntropyLoss()

training_loader = DataLoader(train_dataset, batch_size=val_bsize, shuffle=True,
                             pin_memory=True)
validation_loader = DataLoader(validation_dataset, batch_size=train_bsize,
                               shuffle=False, pin_memory=True)

trainable_params = [p for p in model_des.parameters() if p.requires_grad]
optimizer = optim.Adam(trainable_params, lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
epochs = 10

loss = []       # Per epoch
accuracy = []   # " "

for epoch in range(epochs):

    # Training
    loss_epoch = 0.0
    for batch, (x, y) in enumerate(training_loader):
        x = x.to(device)
        y = y.to(device)
        loss_batch = criterion(model_des(x), y)
        loss_epoch += loss_batch.item()

        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
        scheduler.step()
        print(f"{epoch}.{batch} {loss_batch.item()}")
    loss.append(loss_epoch / len(training_loader))

    # Validation
    with torch.no_grad():
        model_des.eval()
        correct = 0
        for batch, (x, y) in enumerate(validation_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model_des(x).max(dim=1)[1]
            correct += (y_hat == y).sum().item()
        accuracy.append(correct / len(validation_dataset))
        model_des.train()
    print(f"Epoch: {epoch} accuracy: {accuracy[-1]}")

plt.plot(loss)
plt.ylabel("Average loss")
plt.xlabel("epoch")
plt.show()

plt.plot(accuracy)
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()
