import torch.nn as nn

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
