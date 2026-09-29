import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(128, classes)
    def forward(self,x): return self.head(self.features(x).flatten(1))
