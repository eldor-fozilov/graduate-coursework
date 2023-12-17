from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pickle
import torch

class FoodDataset(Dataset):
    def __init__(self, data_path, transform=None):
        
        with open(data_path, 'rb') as file:
            self.data = pickle.load(file)
        
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