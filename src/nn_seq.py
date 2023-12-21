import torch
from torch.nn import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class Tudui(nn.Module):
     def __init__(self):
         super().__init__()
         self.model1 = Sequential(
             Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
             MaxPool2d(kernel_size=2),
             Conv2d(32, 32, 5, padding=2),
             MaxPool2d(2),
             Conv2d(32, 64, 5, padding=2),
             MaxPool2d(2),
             Flatten(),
             Linear(1024, 64),
             Linear(64, 10),
         )

     def forward(self, x):
         x = self.model1(x)
         return x

tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)
writer.close()



