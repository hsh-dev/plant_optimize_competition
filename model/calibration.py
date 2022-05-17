from turtle import forward
import torch.nn as nn
import torch

class CalibrationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1 + 3*15, 512)
        self.bc1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)
        self.bc2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()     
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.bc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x

class CalibrationTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.bc1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(16, 16)
        self.bc2 = nn.BatchNorm1d(16)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x= self.bc2(x)
        x = self.relu2(x)

        x = self.fc3(x)

        return x

def CalibHead():
    return CalibrationHead()

def CalibTail():
    return CalibrationTail()