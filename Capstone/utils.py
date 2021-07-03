"""
Created on Thu Jul  1 13:39:29 2021

@author: gonzr
"""

import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DenominationsData(Dataset):
    """Object representing the denominations dataset."""

    def __init__(self, csv_file, data_dir, transform=None):
        """Create DenomicationsData object.

        :param csv_file: name of csv file
        :type csv_file: str
        :param data_dir: directory where data is
        :type data_dir: str
        :param transform: DESCRIPTION, defaults to None
        :type transform: None, optional

        """
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
        self.len = self.df.shape[0]

    def __len__(self):
        """Return number of samples in the dataset."""
        return self.len

    def __getitem__(self, index):
        """Return X and Y."""
        image_name = self.data_dir + self.df.iloc[index, 2]
        img = Image.open(image_name)
        if self.transform:
            img = self.transform(img)
        label = self.df.iloc[index, -1]

        return img, label


if __name__ == '__main__':

    train_dir = 'data/training/'
    train_csv_file = 'data/training/training_labels.csv'
    train_dataset = DenominationsData(train_csv_file, train_dir)

    validation_dir = 'data/training/'
    validation_csv_file = 'data/validation/validation_labels.csv'
    validation_dataset = DenominationsData(validation_csv_file, validation_dir)

    # Print out the classes for the following samples:
    samples = [53, 23, 10]
    for sample in samples:
        X, Y = train_dataset.__getitem__(sample)
        plt.imshow(X)
        plt.title(f"label: {Y}")
        plt.show()
        print(Y)

    samples = [22, 32, 45]
    for sample in samples:
        X, Y = train_dataset[sample]
        plt.imshow(X)
        plt.title(f"label: {Y}")
        plt.show()
        print(Y)

    # Create a test_normalization dataset to see if the transform is correct.
    # Use the training dataset.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    composed = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])

    test_normalization = DenominationsData(train_csv_file, train_dir,
                                           transform=composed)

    print("Mean: ", test_normalization[0][0].mean(dim=1).mean(dim=1))
    print("Std:", test_normalization[0][0].std(dim=1).std(dim=1))

    # Load and plot samples 0 and 52 from the training data.
    # img = Image.open("data/training/0.jpeg")
    # img.show()
    # img = Image.open("data/training/52.jpeg")
    # img.show()

    # Load and plot samples 1 and 35 from the validation data.
    # validation_dir = "data/validation/"
    # img = Image.open(validation_dir + "1.jpeg")
    # img.show()
    # img = Image.open(validation_dir + "35.jpeg")
    # img.show()

    # Sample number, denomination, file and class variable are in the csv file.
    # train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'
    # train_dir = "data/training/"
    # train_data_name = pd.read_csv(train_csv_file)
    # train_data_name.head()
    # print('File name:', train_data_name.iloc[0, 2])
    # print('y:', train_data_name.iloc[0, 3])
    # print('File name:', train_data_name.iloc[1, 2])
    # print('y:', train_data_name.iloc[1, 3])
    # print('The number of samples (rows): ', train_data_name.shape[0])

    # validation_csv_file = 'https://cocl.us/DL0320EN_VALID_CSV'
    # df = pd.read_csv(train_csv_file)
    # df.head()
    # df.iloc[10, -1:]  # Load the 11th sample image name and class label

    # train_data_dir = 'data/training/'
    # file_name = df.iloc[1, 2]
    # train_image_name = train_data_dir + file_name
    # image = Image.open(train_image_name)
    # plt.imshow(image)

    # validation_data_dir = 'data/validation/'
    # validation_dir = 'data/validation/'
    # validation_image_name = validation_data_dir + file_name
    # image = Image.open(validation_image_name)
    # plt.imshow(image)
