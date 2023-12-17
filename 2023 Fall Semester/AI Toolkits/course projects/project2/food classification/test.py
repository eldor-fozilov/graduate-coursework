
import pickle
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from train_and_test_utils import test
from torchsummary import summary

from PIL import Image
from model import MyModel

from utils import score, load_checkpoint, reset, count_parameters

test_data_path = 'val.pkl'

with open(test_data_path, 'rb') as file:
    test_data = pickle.load(file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256


class TestDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data['path'])

    def __getitem__(self, idx):

        sample = dict()

        img_path = self.data['path'][idx]

        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)


        sample['img'] = image
        sample['id'] = self.data['id'][idx]
        sample['label'] = self.data['label'][idx]

        return sample
    
    
test_transforms = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_dataset = TestDataset(test_data, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

reset(0)

def test(model, sample):
    model.eval()

    with torch.no_grad():

        img = sample['img'].to(device)
        label = sample['id'].to(device)
        pred = model(img)
        num_correct = torch.sum(torch.argmax(pred, dim=-1) == label)

    return num_correct.item()


model = load_checkpoint('checkpoint.pth', device)
#model = MyModel()
#summary(model, (3, 256,256))

num_params = count_parameters(model)
print('Number of parameters: {}'.format(num_params))
if num_params > 20000000:
    raise ValueError("Cannot have more than 20 million parameters!")

avg_te_correct = 0
for sample in test_loader:
    te_correct = test(model, sample)
    avg_te_correct += te_correct / len(test_dataset)

print('Your accuracy: {:.02f}%'.format(avg_te_correct*100))
print('Your score: {:.02f} out of 100'.format(score(avg_te_correct*100)))