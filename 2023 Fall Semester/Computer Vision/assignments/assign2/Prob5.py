import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import DataLoader, TensorDataset
import copy

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
from utils import seed_everything, evaluate_task # , train_task

#* ----------------------- global setup ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 42  
seed_everything(seed_value)

#* ----------------------- hyper params and mdoel ------------------------------
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

NUM_EPOCH = 3 
dataset = MNIST()

original_model = None


# we modified the train_task function
def train_task(model, data_loader, optimizer, criterion, original_model = None, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # Use tqdm for a progress bar
        loop = tqdm(data_loader, total=len(data_loader), leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            if original_model is not None:
                
                with torch.no_grad():
                    # Get the predicted labels from the original model
                    _, original_model_predicted_labels = torch.max(original_model(images).data, 1)
                
                old_loss_coefficient = 1
                # new loss
                loss = criterion(outputs, labels) + old_loss_coefficient * criterion(outputs, original_model_predicted_labels)
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update tqdm progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

        avg_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}] training - Avg Loss: {avg_loss:.6f}, Accuracy: {accuracy:.6f}%")


#* ----------------------- task process ------------------------------
seen_classes_list = []  # Renamed from accumulate
for task_idx, (x_train_task, t_train_task, x_test_task, t_test_task) in enumerate(dataset.task_data):
    # Update seen_classes_list
    seen_classes = np.unique(t_train_task).tolist()
    seen_classes_list.extend(seen_classes)
    seen_classes_list = list(set(seen_classes_list)) # if use the replay method, for preventing ovelapping each classes
    
    # Adapt model's last layer
    model.adapt_last_layer(len(seen_classes_list)) 

    # Convert numpy arrays to PyTorch tensors and ensure they're on the correct device
    x_train_task = torch.tensor(x_train_task).float()
    t_train_task = torch.tensor(t_train_task).long()
 
    train_dataset = TensorDataset(x_train_task, t_train_task)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    #* ----------------------- train ------------------------------
    if task_idx == 0:
        # save the original model
        original_model = copy.deepcopy(model)
        train_task(model, train_loader, optimizer, criterion, None, NUM_EPOCH)
    else:
        train_task(model, train_loader, optimizer, criterion, original_model, NUM_EPOCH)
    
    
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
        
        acc = evaluate_task(model, test_loader, current_task_classes)
        accuracy.append(acc)
    avg_accuracy = sum(accuracy) / len(accuracy)
    print(f"seen classes : {seen_classes_list} \t seen classes acc : {avg_accuracy:.6f}")
    print(f"-" * 50, "\n")
    
    
#* ----------------------- save model ------------------------------
torch.save(model.state_dict(), "Prob5.pth")