import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_c, out_c, stride = 1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel = 3, stride=stride, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels= out_c, out_channels = out_c * BasicBlock.expansion, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_c * BasicBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_c != BasicBlock.expansion * out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_c, out_channels = out_c * BasicBlock.expansion, kernel = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_c * BasicBlock.expansion),
            )
        
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_c, out_c, stride = 1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels = in_c, out_channels = out_c, kernel_size = 1, stride= 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels= out_c, out_channels = out_c, kernel_size=3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels= out_c, out_channels = out_c * BottleNeck.expansion, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_c * BottleNeck.expansion)
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_c != BottleNeck.expansion * out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_c, out_channels = out_c * BottleNeck.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_c * BottleNeck.expansion),
            )
        
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x