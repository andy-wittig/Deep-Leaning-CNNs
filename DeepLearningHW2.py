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

#-------------------- Problem 1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.in_size = 1
        self.out_size = 4
        self.kernel_conv = 3
        self.k_pool = 2
        #Images are 28x28, kernals are 3x3x4, therefore the input to pooling layer is (28-3)/1 + 1 = 26.
        #Pooling divides by 2, so the input to fully connected layer is 13x13x4.
        self.in_fc = 4 * 13 * 13
        self.out_fc = 10 #10 digits possible

        #Convolutional layer with 4 filters of size 3x3.
        self.conv_layer = nn.Conv2d(in_channels = self.in_size, out_channels = self.out_size, kernel_size = self.kernel_conv)
        #Max Pooling layer with size 2x2.
        self.pool_layer = nn.MaxPool2d(kernel_size = self.k_pool)
        self.relu = nn.ReLU()
        self.fully_connected_layer = nn.Linear(in_features = self.in_fc, out_features = self.out_fc)

    def forward(self, output):
        output = self.conv_layer(output)
        output = self.pool_layer(output)
        output = self.relu(output)
        output = output.view(-1, self.in_fc)
        output = self.fully_connected_layer(output)
        return output

#-------------------- Problem 2
model = CNN().to(device)
summary(model, (1, 28, 28))
#There are 6,810 parameters total

#------------------- Training

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Early stopping parameters
early_stopping_patience = 5

# Define the number of epochs to train for
epochs = 20

# Using validation loss as metric
best_val_loss = float('inf')
best_epoch = 0
early_stopping_counter = 0

# Save metrics at each epoch for plotting
epoch_train_loss_values = []
epoch_val_loss_values = []
epoch_train_acc_values = []
epoch_val_acc_values = []

for epoch in range(epochs):
    model.train()  # Set model to training mode

    train_losses, train_accuracies = [], []

    for data, label in train_dl:
        data, label = data.to(device), label.to(device)  # Move data to the same device as the model

        optimizer.zero_grad()  # Clear previous epoch's gradients
        output = model(data)  # Forward pass
        loss = criterion(output, label)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Accumulate metrics
        acc = (output.argmax(dim = 1) == label).float().mean().item()
        train_losses.append(loss.item())
        train_accuracies.append(acc)

    # Average metrics across all training steps
    epoch_train_loss = sum(train_losses) / len(train_losses)
    epoch_train_accuracy = sum(train_accuracies) / len(train_accuracies)

    # Save current epochs training metrics
    epoch_train_loss_values.append(epoch_train_loss)
    epoch_train_acc_values.append(epoch_train_accuracy)

    # Validation
    model.eval()  # Set model to evaluation mode
    val_losses, val_accuracies = [], []
    with torch.no_grad():  # Disable gradient calculation
        for data, label in val_dl:
            data, label = data.to(device), label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            # Accumulate metrics
            acc = (val_output.argmax(dim = 1) == label).float().mean().item()
            val_losses.append(val_loss.item())
            val_accuracies.append(acc)

    # Average metrics across all validation steps
    epoch_val_loss = sum(val_losses) / len(val_losses)
    epoch_val_accuracy = sum(val_accuracies) / len(val_accuracies)

    # Save current epochs validation metrics
    epoch_val_loss_values.append(epoch_val_loss)
    epoch_val_acc_values.append(epoch_val_accuracy)

    # Update best model if validation accuracy improves
    if epoch_val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')

        best_val_loss = epoch_val_loss
        best_epoch = epoch + 1
        early_stopping_counter = 0

    else:
        early_stopping_counter += 1

    print(f'Epoch: {epoch + 1}\n'
          f'Train Acc: {epoch_train_accuracy:.3f}, Val Acc: {epoch_val_accuracy:.3f} '
          f'Train Loss: {epoch_train_loss:.3f}, Val Loss: {epoch_val_loss:.3f}')
    print(f'Best Metric: {best_val_loss:.3f} at epoch: {best_epoch}\n')

    if early_stopping_counter >= early_stopping_patience:
        print(f"Early stopping after {early_stopping_patience} epochs of no improvement.")
        break


# Plot results
plt.figure(figsize = (12, 6))

# Plot loss
plt.subplot(1, 2, 1)
plt.title("Loss")
x_train = [i + 1 for i in range(len(epoch_train_loss_values))]
y_train = epoch_train_loss_values
x_val = [i + 1 for i in range(len(epoch_val_loss_values))]
y_val = epoch_val_loss_values
plt.plot(x_train, y_train, label='Train Loss')
plt.plot(x_val, y_val, label='Val Loss', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.title("Accuracy")
x_train_acc = [i + 1 for i in range(len(epoch_train_acc_values))]
y_train_acc = epoch_train_acc_values
x_val_acc = [i + 1 for i in range(len(epoch_val_acc_values))]
y_val_acc = epoch_val_acc_values
plt.plot(x_train_acc, y_train_acc, label='Train Acc')
plt.plot(x_val_acc, y_val_acc, label='Val Acc', linestyle='--')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()