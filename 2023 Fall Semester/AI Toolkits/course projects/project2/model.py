import torch.nn as nn
import torchvision


class MyModel(nn.Module):
    def __init__(self, dim_output=101):
        super(MyModel, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, dim_output)

    def forward(self, img):
        return self.resnet(img)
