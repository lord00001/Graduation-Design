# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

# 处理单张图片
def Mm(img, img_save_path, img_path):
    # path = './images/'
    # save_path = './test/detect_images/'
    # 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 利用Sobel边缘检测生成二值图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 膨胀、腐蚀
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)
    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)
    #  查找轮廓和筛选文字区域
    region = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算轮廓面积，并筛选掉面积小的
        area = cv2.contourArea(cnt)
        if area < 1000: continue  # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)
        # print("rect is: ")
        # print(rect)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 根据文字特征，筛选那些太细的矩形，留下扁的
        if (height > width * 1.3): continue
        if (height >= width): continue
        if (height < 20 or height > 120): continue
        if (width > 1100): continue
        # if (box[0][0] == box[1][0] or box[0][0] == box[2][0] or box[0][0] == box[3][0]  ):
        region.append(box)
        # else: continue
    # 绘制轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(os.path.join(img_save_path, (os.path.basename(img_path)).split('.')[0]))
    cv2.imwrite(os.path.join(img_save_path, (os.path.basename(img_path)).split('.')[0]+'.jpg'), img)

    # print(len(region))
    bbox = [[] for _ in range(len(region))]
    for i in range(len(region)):
        region[i] = region[i].tolist()
        bbox[i].append(region[i][0][0])
        bbox[i].append(region[i][0][1])
        bbox[i].append(region[i][1][0])
        bbox[i].append(region[i][1][1])
        bbox[i].append(region[i][2][0])
        bbox[i].append(region[i][2][1])
        bbox[i].append(region[i][3][0])
        bbox[i].append(region[i][3][1])
    # print("{}的标注坐标：".format(file.split(".")[0]),bbox)
    # print(len(bbox))
    # print(bbox[0])
    with open('./test/txt/{}.txt'.format((os.path.basename(img_path)).split('.')[0]), 'a+', encoding='utf-8') as f:
        for data in bbox:
            for i in range(len(data)):
                if i != len(data) - 1:
                    f.write(str(data[i]) + ',')
                else:
                    f.write(str(data[i]))
            f.write('\n')
    f.close()

'''
# 批量处理图片
# 读取图片
path = './images/'
savePath = './output_label/'

for file in os.listdir(path):
    imagePath = os.path.join(path, file)
    img = cv2.imread(imagePath)

    # 转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 利用Sobel边缘检测生成二值图
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # 膨胀、腐蚀
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=2)

    #  查找轮廓和筛选文字区域
    region = []
    contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]

        # 计算轮廓面积，并筛选掉面积小的
        area = cv2.contourArea(cnt)
        if area < 1000: continue  # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)
        # print("rect is: ")
        # print(rect)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        # 根据文字特征，筛选那些太细的矩形，留下扁的
        if (height > width * 1.3): continue
        if (height >= width): continue
        if (height < 20 or height > 120): continue
        if (width > 1100): continue
        # if (box[0][0] == box[1][0] or box[0][0] == box[2][0] or box[0][0] == box[3][0]  ):
        print(width)
        region.append(box)
        # else: continue


    # 绘制轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    save_path = os.path.join(savePath, file)
    cv2.imwrite(save_path, img)

    # print(len(region))
    bbox = [[] for _ in range(len(region))]
    for i in range(len(region)):
        region[i] = region[i].tolist()
        bbox[i].append(region[i][0][0])
        bbox[i].append(region[i][0][1])
        bbox[i].append(region[i][1][0])
        bbox[i].append(region[i][1][1])
        bbox[i].append(region[i][2][0])
        bbox[i].append(region[i][2][1])
        bbox[i].append(region[i][3][0])
        bbox[i].append(region[i][3][1])
    # print("{}的标注坐标：".format(file.split(".")[0]),bbox)
    # print(len(bbox))
    # print(bbox[0])
    with open('./output_txt/{}.txt'.format(file.split('.')[0]), 'a+', encoding='utf-8') as f:
        for data in bbox:
            for i in range(len(data)):
                if i != len(data)-1:
                   f.write(str(data[i])+',')
                else:
                   f.write(str(data[i]))
            f.write('\n')
    f.close()
'''
