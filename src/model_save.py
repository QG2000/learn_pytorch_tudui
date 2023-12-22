import torch
import torchvision

vgg16 = torchvision.models.vgg16(weights=None)

#保存方式１
torch.save(vgg16, "vgg16_method1.pth")

#保存方式２
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
