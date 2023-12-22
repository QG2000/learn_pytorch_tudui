import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
# from model import *

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


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

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=4 * 4 * 64, out_features=64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 损失函数－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(params=tudui.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
epoch = 10

writer = SummaryWriter("../logs_train")
start_time = time.time()
for i in range(epoch):
    print("--------------------------第　{}　轮训练开始-------------------------------------------".format(i+1))

    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数：{}, 损失是: {}".format(total_train_step, loss.item()))
            print(end_time - start_time)
            writer.add_scalar("train_loss", loss, total_train_step)

# －－－－－－－－－－－－－－－－－下面是测试－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

        print("整体测试集上的ｌｏｓｓ:{}".format(total_test_loss))
        print("---整体测试集上的正确率：--------{}-----".format(total_accuracy/test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
        total_test_step = total_test_step + 1

    torch.save(tudui, "tudui_{}.pth".format(i))
    print("＝＝＝＝＝＝＝＝＝＝模型以保存＝＝＝＝＝＝＝＝＝＝")

writer.close()



