import os
import sys
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from PIL import Image

from model import MyModel
from utils import score, load_checkpoint, reset, count_parameters


data_fp = 'val.npz'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64

class TestDataset(Dataset):
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
    


test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.286], std=[0.353]),
    ]
)

test_dataset = TestDataset(data_fp, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


def test(model, sample):
    model.eval()

    with torch.no_grad():
        input, label = sample[0].to(device), sample[1].to(device)
        pred = model(input)
        num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)

    return num_correct.item()

reset(0)
model = load_checkpoint('checkpoint.pth', device)

num_params = count_parameters(model)
if num_params > 1000000:
    raise ValueError("Cannot have more than 1 million parameters!")

avg_te_correct = 0
for sample in test_loader:
    te_correct = test(model, sample)
    avg_te_correct += te_correct / len(test_dataset)

print('Your accuracy: {:.02f}%'.format(avg_te_correct*100))
print('Your score: {:.02f} out of 100'.format(score(avg_te_correct*100)))