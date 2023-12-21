import torch
from torch.nn import *
from torch import nn

class Tudui(nn.Module):
     def __init__(self):
         super().__init__()
         self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
         self.maxpool1 = MaxPool2d(kernel_size=2)
         self.conv2 = Conv2d(32, 32, 5, padding=2)
         self.maxpool2 = MaxPool2d(2)
         self.conv3 = Conv2d(32, 64, 5, padding=2)
         self.maxpool3 = MaxPool2d(2)
         self.flatten = Flatten()
         self.linear1 = Linear(1024, 64)
         self.linear2 = Linear(64, 10)

     def forward(self, x):
         x = self.conv1(x)
         x = self.maxpool1(x)
         x = self.conv2(x)
         x = self.maxpool2(x)
         x = self.conv3(x)
         x = self.maxpool3(x)
         x = self.flatten(x)
         x = self.linear1(x)
         x = self.linear2(x)
         return x

tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)




