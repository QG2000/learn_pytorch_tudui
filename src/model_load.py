import torch

#加载方式一
model = torch.load("vgg16_method1.pth")
# print(model)


#加载方式２
model2 = torch.load("vgg16_method2.pth")
print(model2)
