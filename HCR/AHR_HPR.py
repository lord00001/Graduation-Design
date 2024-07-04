import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import transforms, utils,datasets
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 构建CNN模型
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

cnn_model = CNNNet()

def deal_img(image):
    """Transforming images on GPU"""
    image_new =  cv2.resize(image, (224,224))
    # image_new=image
    image_new = Image.fromarray(image_new)  # 这里ndarray_image为原来的numpy数组类型的输入
    my_transforms= transforms.Compose([
        transforms.Resize(32), # 缩放图片(Image)，保持长宽比不变，最短边为32像素
        transforms.CenterCrop(32), # 从图片中间切出32*32的图片
        transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
        transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251]) # 标准化至[-1, 1]，规定均值和标准差
    ])
    my_tensor = my_transforms(image_new)
    my_tensor = my_tensor.unsqueeze(0)
    my_tensor= my_tensor.cuda()
    return my_tensor

def cls_inference(cls_model,img):
    input_tensor = deal_img(img)

    cls_model.cuda()
    cls_model.eval()
    result = cls_model(input_tensor)
    result_npy = result.data.cpu().numpy()
    max_index = np.argmax(result_npy[0])
    return max_index

# 签名 / 合格
def AHR(model_path, img):
    # 加载训练好的模型
    cnn_model = torch.load(model_path)
    cnn_model.eval()
    model = cnn_model
    AH_label = cls_inference(model, img)
    if AH_label == 0:
        print('签名')
    if AH_label == 1:
        print('合格')
    return AH_label

# 手写字 / 印刷字
def HPR(model_path, img):
    # 加载训练好的模型
    cnn_model = torch.load(model_path)
    cnn_model.eval()
    model = cnn_model
    HP_label = cls_inference(model, img)
    if HP_label == 0:
        print('手写')
    if HP_label == 1:
        print('印刷')
    return HP_label

AH_model_path = './HCR/model/HAR_cnn_model.pt'
HP_model_path = './HCR/model/HPR_cnn_model.pt'

if __name__ == "__main__":
    image = cv2.imread('./data/AHR_data/test/autograph/autograph1755.jpg')
    AH_model_path = './model/HAR_cnn_model.pt'
    HP_model_path = './model/HPR_cnn_model.pt'
    AHR(AH_model_path, image)
    HPR(HP_model_path, image)