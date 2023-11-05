import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.model(x)
