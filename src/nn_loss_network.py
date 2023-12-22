import torch
import torchvision.datasets
from torch.nn import *
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1 )

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
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    result_loss = loss(output, targets)
    print(result_loss)
    result_loss.backward()
    print('ok')

    
