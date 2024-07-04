import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils,datasets
import cv2
from PIL import Image
from model import *

# 参数设置
num_epochs = 15
batch_size = 64
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 图片显示
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# torchvision 数据集的输出是范围在[0,1]之间的 PILImage，我们将他们转换成归一化范围为[-1,1]之间的张量Tensors
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# # 获取CIFAR10训练集和测试集
# trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform)
# testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transform)

data_transform = transforms.Compose([
 transforms.Resize(32), # 缩放图片(Image)，保持长宽比不变，最短边为32像素
 transforms.CenterCrop(32), # 从图片中间切出32*32的图片
 transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
 transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251]) # 标准化至[-1, 1]，规定均值和标准差
])
trainset = datasets.ImageFolder(root="C:/Users/龙耀九州/Desktop/市创终期/data3/train",
           transform=data_transform) #导入数据集
# img, label = train_dataset[3100] #将启动魔法方法__getitem__(0)
# print(label)   #查看标签
# print(img.size())
# print(img)
#
# #处理后的图片信息
# for img, label in train_dataset:
#  print("图像img的形状{},标签label的值{}".format(img.shape, label))
#  print("图像数据预处理后：\n",img)
#  break
testset = datasets.ImageFolder(root="C:/Users/龙耀九州/Desktop/市创终期/data3/test",
           transform=data_transform) #导入数据集



# CIFAR10训练集和测试集装载
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
# 图片类别
classes = ('shouxie', 'yinshua')
# 图片显示
images, labels = next(iter(trainloader))
imshow(torchvision.utils.make_grid(images))

# 定义损失函数和优化器
cnn_model = CNNNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=learning_rate, momentum=0.9)



# # 训练模型
for epoch in range(num_epochs):
    running_loss = 0.00
    running_correct = 0.0
    print("Epoch  {}/{}".format(epoch, num_epochs))
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        running_correct += torch.sum(pred == labels.data)
    print("Loss is :{:.4f},Train Accuracy is:{:.4f}%".format(running_loss / len(trainset),
                                                             100 * running_correct / len(trainset)))
# 保存训练好的模型
torch.save(cnn_model, 'data3/cnn_model.pt')

# 加载训练好的模型
cnn_model = torch.load('data3/cnn_model.pt')
cnn_model.eval()
# 使用测试集对模型进行评估
correct = 0.0
total = 0.0
with torch.no_grad():  # 为了使下面的计算图不占用内存
    for data in testloader:
        images, labels = data
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print("Test Average accuracy is:{:.4f}%".format(100 * correct / total))

# 求出每个类别的准确率
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        try:
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        except IndexError:
            continue
for i in range(2):
    print('Accuracy of %5s : %4f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

imagepath="C:/Users/龙耀九州/Desktop/市创终期/picture"
testset2 = datasets.ImageFolder(root=imagepath,
           transform=data_transform) #导入数据集
testloader2 = torch.utils.data.DataLoader(testset2,shuffle=False, num_workers=0)


