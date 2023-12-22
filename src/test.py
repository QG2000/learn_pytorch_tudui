import torch
import torchvision.transforms
from PIL import Image
from torch import nn

image_path = "../imgs/dog.png"
image = Image.open(image_path)
image = image.convert('RGB')
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)

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


model = torch.load("tudui_9.pth")
print(model)
image = image.to('cuda')  # 将输入数据移动到 GPU 上
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax())

