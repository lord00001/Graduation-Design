# --- coding：UTF-8 ---
from skimage import io, data, morphology, segmentation, color
import numpy as np
from PIL import Image
import pytesseract
import os
import xlwt
import matplotlib.pyplot as plt
import cv2
from data_preprocess.data_preprocess import *

# pytesser3.tesseract_exe_name="C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"
# 识别图片中的文字
def img_to_txt(file):
    if os.path.exists(file):
        image = Image.open(file)
        # 英文
        # vcode = pytesseract.image_to_string(image,"eng")
        # txt = pytesser3.image_to_string(image, "chi_sim")
        txt = pytesseract.image_to_string(image, "chi_sim")
        print("{}识别内容:{}".format(file, txt))


# for i in range(16):
#     img_file = r"img/pic_{}.jpg".format(i)
#     img_to_txt(img_file)


# retVal, a_img = cv2.threshold(img2, 0, 255, cv2.THRESH_OTSU)

def OTSU(img_gray, GrayScale):
    assert img_gray.ndim == 2, "must input a gary_img"  # shape有几个数字, ndim就是多少
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    u1 = 0.0  # 背景像素的平均灰度值
    u2 = 0.0  # 前景像素的平均灰度值
    th = 0.0

    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例。
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for i in range(1, GrayScale):  # 从1开始是为了避免w1为0.
        u1_tem = 0.0
        u2_tem = 0.0
        # 背景像素的比列
        w1 = np.sum(PixRate[:i])
        # 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:  # 背景像素的平均灰度值
            for m in range(i):
                u1_tem = u1_tem + PixRate[m] * m
            u1 = u1_tem * 1.0 / w1
            # 前景像素的平均灰度值
            for n in range(i, GrayScale):
                u2_tem = u2_tem + PixRate[n] * n
            u2 = u2_tem / w2
            # print(u1)
            # 类间方差公式：G=w1*w2*(u1-u2)**2
            tem_var = w1 * w2 * np.power((u1 - u2), 2)
            # print(tem_var)
            # 判断当前类间方差是否为最大值。
            if Max_var < tem_var:
                Max_var = tem_var  # 深拷贝，Max_var与tem_var占用不同的内存空间。
                th = i
    return th


def dil2ero(img, selem):
    img = morphology.dilation(img, selem)  # 膨胀
    imgres = morphology.erosion(img, selem)  # 腐蚀
    return imgres

# io.imsave("./temp/table_dot.jpg", img_dot)


def isolate(imgdot):
    idx = np.argwhere(imgdot < 1)  # img值小于1的索引数组
    rows, cols = imgdot.shape

    for i in range(idx.shape[0]):
        c_row = idx[i, 0]
        c_col = idx[i, 1]
        if c_col + 1 < cols and c_row + 1 < rows:
            imgdot[c_row, c_col + 1] = 1
            imgdot[c_row + 1, c_col] = 1
            imgdot[c_row + 1, c_col + 1] = 1
        if c_col + 2 < cols and c_row + 2 < rows:
            imgdot[c_row + 1, c_col + 2] = 1
            imgdot[c_row + 2, c_col] = 1
            imgdot[c_row, c_col + 2] = 1
            imgdot[c_row + 2, c_col + 1] = 1
            imgdot[c_row + 2, c_col + 2] = 1
    return imgdot


def clearEdge(img, width):
    img[0:width - 1, :] = 1
    img[1 - width:-1, :] = 1
    img[:, 0:width - 1] = 1
    img[:, 1 - width:-1] = 1
    return img


# io.imsave("./temp/table_dot_del.jpg", img_dot)


# print(img_dot.shape)  # 显示尺寸
# print(img_dot.shape[0])  # 图片高度
# print(img_dot.shape[1])  # 图片宽度


def average(list):
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)


def split_table(list, list0):
    table_frame = []
    for i in list0:
        table_frame.append([i])
    # print(table_frame)
    for i in range(len(list0)):
        for j in range(len(list)):
            if abs(list[j][0]-list0[i][0]) <= 16 and abs(list[j][1] - list0[i][1]) >= 2:
                table_frame[i].append(list[j])
    for l in range(len(table_frame)):
        table_frame[l].sort(key=lambda x: x[1])
    print("table_frame: ", table_frame)
    # print(table_frame[24])
    # print("length of table_frame: ", len(table_frame))
    return table_frame


def leftPoint_process(list):
    list_y, list_x, list0, list1, list2, list3, list4, list5, list6, list7 = [], [], [], [], [], [], [], [], [], []
    list.sort(key=lambda x: x[1])
    # print(list)
    for i in list:                      # 得到每一列的坐标，即分类
        if i[1] in range(85, 185):
            list0.append(i)
        # elif i[1] in range(220, 315):
        #     list1.append(i)
        # elif i[1] in range(316, 400):
        #     list2.append(i)
        # elif i[1] in range(570, 700):
        #     list3.append(i)
        # elif i[1] in range(730, 885):
        #     list4.append(i)
        # elif i[1] in range(980, 1150):
        #     list5.append(i)
        # elif i[1] in range(1160, 1350):
        #     list6.append(i)
        elif i[1] in range(1362, 1560):
            list7.append(i)
    list0.sort(key=lambda x: x[0])
    list7.sort(key=lambda x: x[0])

    # if len(list0) > 37:
    #     for i in range(len(list0) - 1):
    #         if abs(list0[i][0] - list0[i+1][0]) <=5:
    #             list0.remove(list0[i])
    # print(len(list0))
    # print(len(list7))
    # print("list0: ", list0)
    # print("list7: ", list7)
    if len(list0) < len(list7) <= 37:
        for i in list0:
            list_x.append(i[1])
        x = average(list_x)
        for j in range(len(list7)):
            # print("j", j)
            if list0[j][0] - list7[j][0] >= 15:  # 用纵坐标的阈值衡量两个坐标点是否在一行
                list0.insert(j, [list7[j][0], int(x)])
                list.append([list7[j][0], int(x)])

        # print(len(list0))
        # print(len(list7))

    if 37 >= len(list0) > len(list7):
        for i in list7:
            list_x.append(i[1])
        x = average(list_x)
        # print('len(list0)', len(list0))
        for j in range(len(list0)):
            print(j)
            if abs(list7[j][0] - list0[j][0]) >= 15:
                list7.insert(j, [list0[j][0], int(x)])
                print(len(list7))
                list.append([list0[j][0], int(x)])

    # print("list: ", list)
    table_frame = split_table(list, list0)
    return table_frame


class goubi:
    def __init__(self, img_dot, w, h):
        self.img_dot = img_dot
        self.w = w
        self.h = h

    # 表格点图
    def get_dot(self):
        self.img_dot2 = self.isolate(self.img_dot)
        # io.imsave("./test/temp/table_dot_del.jpg", self.img_dot2)
        return self.img_dot2

    # 已知三个点，求第四个点的坐标 coor = [纵坐标， 横坐标]

    # 分析表格
    def fenxi_dots(self):
        self.dot_idxs = np.argwhere(self.img_dot < 1)  # img_dot值等于0的索引数组
        self.table_cols = []  # 记录每行有几个单元格

        temp = self.dot_idxs.tolist()
        #  清除多余图表外定格点
        index_to_delete = []
        for i in range(len(temp)):
            if temp[i][1] <= self.w/20 or temp[i][1] >= self.w*12/13:
                index_to_delete.append(i)
            if temp[i][0] > 18* self.h/20 or temp[i][0] < self.h/20:
                index_to_delete.append(i)
        temp = [temp[i] for i in range(len(temp)) if i not in index_to_delete]
        print(temp)
        table_frame = leftPoint_process(temp)

        return table_frame

        # return temp

def layout_analysis_to_frame(imgpath):

    img = io.imread(imgpath, True)  # as_grey=true 灰度图片

    img2 = cv2.imread(imgpath, 0)  # as_grey=true 灰度图片

    size = img2.shape
    w = size[1]  # 宽度
    h = size[0]  # 高度
    th = OTSU(img2, 256)
    bi_th = th / 256 + 0.15
    print(bi_th)
    # 二值化
    # bi_th = 0.6

    img[img <= bi_th] = 0
    img[img > bi_th] = 1

    # io.imsave("./test/temp/gray.jpg", img)
    rows, cols = img.shape
    scale = 30  # 这个值越大,检测到的直线越多,图片中字体越大，值应设置越小。需要灵活调整该参数。

    col_selem = morphology.rectangle(cols // scale, 1)
    img_cols = dil2ero(img, col_selem)

    row_selem = morphology.rectangle(1, rows // scale)
    img_rows = dil2ero(img, row_selem)

    img_line = img_cols * img_rows
    # io.imsave("./test/temp/table.jpg", img_line)

    img_dot = img_cols + img_rows
    img_dot[img_dot > 0] = 1
    img_dot = isolate(img_dot)
    list = img_dot.tolist()
    G = goubi(img_dot, w, h)
    frame = G.fenxi_dots()
    return frame
