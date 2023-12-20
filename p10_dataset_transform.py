<<<<<<< 70e3ab1fc6c9b260147d172da7826e59ec63e697
=======
import torchvision

tran_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)

print(test_set[0])
>>>>>>> 1
