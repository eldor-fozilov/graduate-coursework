import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset

# some initial imports
import random
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
from model import Net
from utils import seed_everything, evaluate_task, train_task

#* ----------------------- global setup ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 42  
seed_everything(seed_value)

def transfer_learning(model, output_size):
    # freeze the backbone CNN
    for param in model.parameters():
        param.requires_grad = False

    # save the old layer
    old_fc2 = model.fc2
    # new layer
    new_fc2 = nn.Linear(model.fc2.in_features, output_size).to(device)

    return old_fc2, new_fc2


#* ----------------------- hyper params and mdoel ------------------------------



model = Net().to(device)
criterion = nn.CrossEntropyLoss()

NUM_EPOCH = 3
dataset = MNIST()

#* ----------------------- task process ------------------------------
seen_classes_list = []  # Renamed from accumulate
for task_idx, (x_train_task, t_train_task, x_test_task, t_test_task) in enumerate(dataset.task_data):
    # Update seen_classes_list
    seen_classes = np.unique(t_train_task).tolist()
    seen_classes_list.extend(seen_classes)
    seen_classes_list = list(set(seen_classes_list)) # if use the replay method, for preventing ovelapping each classes
    
    
    if task_idx == 0:
        # Adapt model's last layer
        model.adapt_last_layer(len(seen_classes_list))
    if task_idx == 1:
        # for second task add a new layer and freeze the backbone CNN
        old_fc2, new_fc2 = transfer_learning(model, len(seen_classes_list))
        model.fc2 = new_fc2

    # Convert numpy arrays to PyTorch tensors and ensure they're on the correct device
    x_train_task = torch.tensor(x_train_task).float()
    t_train_task = torch.tensor(t_train_task).long()
    train_dataset = TensorDataset(x_train_task, t_train_task)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    #* ----------------------- train ------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_task(model, train_loader, optimizer, criterion, NUM_EPOCH)
    
    #* ----------------------- eval ------------------------------
    print(f"\n -----------------------evaluation start-----------------------")
    accuracy = []
    for task_i, (_, _, x_test, t_test) in enumerate(dataset.task_data):
        if task_i > task_idx :
            continue
        
        current_task_classes = np.unique(t_test).tolist()
        x_test = torch.tensor(x_test).float()
        t_test = torch.tensor(t_test).long()
        test_data = TensorDataset(x_test, t_test)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        if task_idx == 1:
          split_idx = len(old_fc2.weight.data)
          model.fc2.weight.data[:split_idx] = old_fc2.weight.data
          model.fc2.bias.data[:split_idx] = old_fc2.bias.data
        
        acc = evaluate_task(model, test_loader, current_task_classes)
        accuracy.append(acc)
    
    avg_accuracy = sum(accuracy) / len(accuracy)
    print(f"seen classes : {seen_classes_list} \t seen classes acc : {avg_accuracy:.6f}")
    print(f"-" * 50, "\n")
    

#* ----------------------- save model ------------------------------
torch.save(model.state_dict(), "Prob2.pth")