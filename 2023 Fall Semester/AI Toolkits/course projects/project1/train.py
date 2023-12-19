
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from utils import score, load_checkpoint, reset, count_parameters


# Define the Dataset Class and DataLoader"""

class FashionMNISTDataset(Dataset):
    def __init__(self, npz_fp, transform=None):
        with np.load(npz_fp, allow_pickle=True) as data:
            self.data = data["data"]
            self.labels = data["labels"]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_data = self.data[idx].astype("uint8").reshape((28, 28))
        img_label = int(self.labels[idx])

        img_data = Image.fromarray(img_data)

        if self.transform:
            img_data = self.transform(img_data)

        return img_data, img_label

train_data_fp = 'train.npz'
val_data_fp = 'val.npz'
batch_size = 256

train_transforms = transforms.Compose(
    [
        transforms.TrivialAugmentWide(), # transforms.RandAugment()
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286], std=[0.353]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286], std=[0.353]),
    ]
)

train_dataset = FashionMNISTDataset(train_data_fp, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = FashionMNISTDataset(val_data_fp, transform=test_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class MyModel(nn.Module):
  def __init__(self, channels=[32, 64, 128, 256], dim_output=10):
     super().__init__()
     self.channels = channels
     self.block1 = nn.Sequential(
            nn.Conv2d(1, self.channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
     self.block2 = nn.Sequential(
            nn.Conv2d(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(self.channels[2], self.channels[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
     self.block3 = nn.Sequential(nn.Conv2d(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))
     self.ff_network = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256*3*3, 156),
            nn.BatchNorm1d(156),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(156, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, dim_output))

  def forward(self, img):
    out = self.block1(img)
    out = self.block2(out)
    out = self.block3(out)
    out = self.ff_network(out)
    return out

from torchsummary import summary

model = MyModel()
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

dummy_input = torch.rand(batch_size, 1, 28, 28).to(device)
dummy_output = model(dummy_input)
summary(model, (1,28,28))

def train(model, optimizer, sample):
    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad()

    img = sample[0].float().to(device)
    label = sample[1].long().to(device)

    pred = model(img)

    num_correct = sum(torch.argmax(pred, dim=-1) == label)

    pred_loss = criterion(pred, label)

    pred_loss.backward()

    optimizer.step()

    return pred_loss.item(), num_correct.item()

def test(model, sample):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        img = sample[0].float().to(device)
        label = sample[1].long().to(device)

        pred = model(img)
        pred_loss = criterion(pred, label)

        num_correct = sum(torch.argmax(pred, dim=-1) == label)

    return pred_loss.item(), num_correct.item()

max_epoch = 1000

for epoch in tqdm(range(295, max_epoch +1)):
    ###Train Phase

    # Initialize Loss and Accuracy
    train_loss = 0.0
    train_accu = 0.0

    # Iterate over the train_dataloader
    for idx, sample in enumerate(train_loader):
        curr_loss, num_correct = train(model, optimizer, sample)
        train_loss += curr_loss / len(train_loader)
        train_accu += num_correct / len(train_dataset)

    ### Test Phase
    # Initialize Loss and Accuracy
    test_loss = 0.0
    test_accu = 0.0

    # Iterate over the test_dataloader
    for idx, sample in enumerate(val_loader):
        curr_loss, num_correct = test(model, sample)
        test_loss += curr_loss / len(val_loader)
        test_accu += num_correct / len(val_dataset)
        
    checkpoint = {'model': MyModel(), 'model_state_dict': model.state_dict()}
    torch.save(checkpoint, f'./checkpoints/model_after_{epoch}_epochs.pth')
    print('[EPOCH {}] TR LOSS : {:.03f}, TE LOSS :{:.03f}, TR ACCU: {:.03f}, TE ACCU :{:.03f}'.format(epoch, train_loss, test_loss, train_accu, test_accu))

