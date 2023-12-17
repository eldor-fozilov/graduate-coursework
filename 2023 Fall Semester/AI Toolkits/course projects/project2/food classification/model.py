import torch.nn as nn
import torchvision


"""Due to the constraint of 20 million parameters, we used parameter-efficient pretrained model called EfficientNet (B4 version)
with approximately 19 million parameters as our backbone, which is based on CNN architecture."""

class MyModel(nn.Module):
    def __init__(self, dim_output=101, freeze_backbone=False):
        super(MyModel, self).__init__()
        
        self.model = torchvision.models.efficientnet_b4(weights = "EfficientNet_B4_Weights.DEFAULT")
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1792, 512), nn.Dropout(0.2), nn.ReLU(), nn.Linear(512, dim_output)) # 1792 is the output of the last layer of the backbone
    def forward(self, img):        
        output = self.model(img)
        return output
