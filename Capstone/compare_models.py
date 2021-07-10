"""Compare the model performance between ResNet18 and Densenet121.

Created on Sat Jul 10 10:39:36 2021

@author: gonzr
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import resnet18, densenet121
from torchvision import transforms
from tqdm import tqdm
from utils import DenominationsData


def test(model, loader, device="cpu"):
    """Accuracy given current model parameters."""
    with torch.no_grad():
        model.eval()
        num_samples, correct = 0, 0
        for batch, (x, y) in tqdm(enumerate(loader)):
            x = x.to(device)
            y = y.to(device)
            num_samples += y.numel()
            y_hat = model(x).max(dim=1)[1]
            correct += (y_hat == y).sum().item()
        model.train()
    return correct / num_samples


device = "cuda:0"


##############################################################################
#                                                                            #
#                                     Data                                   #
#                                                                            #
##############################################################################

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])

test_dir = 'data/test/'
test_csv_file = 'data/test/test_labels.csv'
test_dataset = DenominationsData(test_csv_file, test_dir, transform=composed)
test_loader = DataLoader(test_dataset, batch_size=15,
                         shuffle=False, pin_memory=True)


##############################################################################
#                                                                            #
#                                    Models                                  #
#                                                                            #
##############################################################################

model_resnet18 = resnet18()
model_resnet18.fc = nn.Linear(in_features=512, out_features=7)
model_resnet18.load_state_dict(torch.load("resnet18.pt"))
model_resnet18 = model_resnet18.to(device)

model_densenet121 = densenet121()
model_densenet121.classifier = nn.Linear(in_features=1024, out_features=7)
model_densenet121.load_state_dict(torch.load("densenet121.pt"))
model_densenet121 = model_densenet121.to(device)


##############################################################################
#                                                                            #
#                                    Test                                    #
#                                                                            #
##############################################################################

accuracy_resnet18 = test(model_resnet18, test_loader, device=device)
accuracy_densenet121 = test(model_densenet121, test_loader, device=device)

print(f"Accuracy on test set for ResNet18: {accuracy_resnet18}")
print(f"Accuracy on test set for DenseNet121: {accuracy_densenet121}")
