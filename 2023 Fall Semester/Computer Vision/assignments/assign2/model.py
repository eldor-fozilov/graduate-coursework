import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset

# some initial imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from cl_dataset import ContinualMNIST as MNIST
from einops import rearrange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        '''
            Initialize the basic architecture. 
        '''
        super(Net, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Dropout
        self.conv_drop = nn.Dropout2d(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.conv_drop(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def adapt_last_layer(self, output_size):
        """Dynamically adapt the last layer for the given output size and retain previously learned weights."""
        if output_size == self.fc2.out_features:
            return
        
        # Step 1: Store the current FC layer's weights and biases
        current_weights = self.fc2.weight.data
        current_biases = self.fc2.bias.data

        # Step 2: Create a new FC layer with the updated size
        new_fc = nn.Linear(self.fc2.in_features, output_size).to(device)
        
        # Step 3: Copy the weights and biases from the old layer to the new layer
        new_fc.weight.data[:len(current_biases)] = current_weights
        new_fc.bias.data[:len(current_biases)] = current_biases
        
        # Assign the new FC layer to the model
        self.fc2 = new_fc
        print(f"extend model layers as a incremental classes in each task")