import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from model import *

train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度: {}".format(train_data_size))
print("测试数据集的长度: {}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

tudui = Tudui()

loss_fn = nn.CrossEntropyLoss()

learning_rate = 1e-2
optimizer = torch.optim.SGD(params=tudui.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
epoch = 10

for i in range(epoch):
    print("--------------------------第　{}　轮训练开始-------------------------------------------".format(i+1))

    for data in train_dataloader:
        imgs, targets = data
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        print("训练次数：{}, 损失是: {}".format(total_train_step, loss.item()))
