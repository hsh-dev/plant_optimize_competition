from turtle import forward
import torch.nn as nn
import torch

class CalibrationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048 + 54, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 16)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CalibrationTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(17, 17)
        self.fc2 = nn.Linear(17, 17)
        self.fc3 = nn.Linear(17, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
