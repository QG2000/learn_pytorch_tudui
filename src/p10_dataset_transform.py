import torchvision
from torch.utils.tensorboard import SummaryWriter

datasets_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

tran_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=datasets_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=datasets_transform, download=True)

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

# print(test_set[0])

writer = SummaryWriter("../p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()
