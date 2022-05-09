import torch
import torch.nn as nn
import torch.nn.functional as F

from model.residual_block import BasicBlock, BottleNeck

class ResNet(nn.Module):
    def __init__(self, block, num_block, init_weights = True):
        super().__init__()

        self.in_channels = 64
        
        # 3x128x128 -> 64x56x56
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )

        # 64x56x56 -> 256x56x56
        self.conv2 = self.make_layer(block, 64, num_block[0], 1)

        # 256x56x56 -> 512x28x28
        self.conv3 = self.make_layer(block, 128, num_block[1], 2)
        
        # 512x28x28 -> 1024x14x14
        self.conv4 = self.make_layer(block, 256, num_block[2], 2)
        
        # 1024x14x14 -> 2048x7x7
        self.conv5 = self.make_layer(block, 512, num_block[3], 2)

        # 2048x7x7 -> 2048x1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # 2048 -> 512
        self.fc1 = nn.Linear(512 * block.expansion, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks-1)
        for stride_ in strides:
            layers.append(block(self.in_channels, out_channels, stride_))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)

        return x

class Tail(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def ResNetTail():
    return Tail()
