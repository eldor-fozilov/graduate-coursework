import torch
import torch.nn as nn
import torchvision

class CNN_Model(nn.Module):
    def __init__(self, num_classes_hour = 12, num_classes_minute = 60, freeze_backbone = False):
        
        super().__init__()
        
        # use ResNet18 as the base model
        self.resnet = torchvision.models.resnet18(weights = "ResNet18_Weights.DEFAULT")

        # freeze the parameters of backbone
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # modify the fully connected layers
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
            
        # add custom layers for predicting hour and minute
        self.hour_fc = nn.Linear(in_features, num_classes_hour)
        self.minute_fc = nn.Linear(in_features, num_classes_minute)

    def forward(self, x):
        x = self.resnet(x)
        hour_output = self.hour_fc(x)
        minute_output = self.minute_fc(x)
        return hour_output, minute_output
    