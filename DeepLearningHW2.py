#Imports for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchsummary import summary

#Imports for vision tasks
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

#Imports for preparing dataset
import os
import numpy as np

#Imports for visualizations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#-------------------- Import and Prepare Dataset

from torchvision.datasets import MNIST

train_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to a FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
])

val_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to a FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
])

# Download and load the training data
full_train_data = datasets.MNIST(root='/tmp', train=True, download=True, transform=train_transform)
full_val_data = datasets.MNIST(root='/tmp', train=False, download=True, transform=val_transform)

# Get indices for subset of full dataset
train_data_size = 1000
val_data_size = 200

train_indices = np.random.rand(train_data_size) * len(full_train_data)
train_indices = np.floor(train_indices).astype(int)
val_indices = np.random.rand(val_data_size) * len(full_val_data)
val_indices = np.floor(val_indices).astype(int)

# Take subset of full dataset
train_data = Subset(full_train_data, indices=train_indices)
val_data = Subset(full_val_data, indices=val_indices)

# Experiment with different batch sizes, generally use 2^x batch size
batch_size = 32
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

#-------------------- Visualize Dataset

# Just to check if the dataset is loaded correctly, let's print the size of the train and test datasets
print(f"Training dataset size: {len(train_data)}")
print(f"Validation dataset size: {len(val_data)}")

# Useful attributes for understanding the dataset
print('Classes in the dataset:', train_data.dataset.classes)  # need to access classes inside of the dataset attribute since we used Subset in the block above

img, label = train_data[0]
print('Image dimensions:', img.shape)

# Define number of classes for model
num_classes = len(train_data.dataset.classes)

def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (12,9))
        ax.set_xticks([])
        ax.set_yticks([])
        images = images.clamp(0, 1)
        ax.imshow(make_grid(images,nrow=8).permute(1,2,0))
        plt.show()
        break

print('\nExample batch of images:')
show_batch(train_dl)

#-------------------- Set Device

# Set device to GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#--------------------

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #Convolutional layer with 4 filters of size 3x3.
        self.conv_layer = nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3)
        #Max Pooling layer with size 2x2.
        self.pool_layer = nn.MaxPool2d(kernel_size = 2)
        self.relu = nn.ReLU()
        #Images are 28x28, kernals are 3x3x4, therefore the input to pooling layer is (28-3)/1 + 1 = 26.
        #Pooling divides by 2, so the input to fully connected layer is 13x13x4.
        self.fully_connected_layer = nn.Linear(in_features = 4 * 13 * 13, out_features = 10) #10 digits possible

    def forward(self):
        #forward processing
