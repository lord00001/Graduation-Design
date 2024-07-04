import os
import cv2
from PIL import Image
import numpy as np

''' 输入： 彩色图片
    输出： 识别的文字  '''

''' 输入： 彩色图片
    输出： 标注的图片  '''

''' 输入： 彩色图片
    输出： 识别的文字 和 检测出的四个点坐标  '''

''' 输入： 彩色图片 和 四个点的坐标
    输出： 检测区域照片list  '''

# 根据四个点的坐标计算其（x，y，x+w，y+h），便于后续裁剪
def calcu_crop_para(coordinate):  # (x1:0, y1:1); (x2:2, y2:3); (x3:4, y3:5); (x4:6, y4:7)
    x_min = min(coordinate[0], coordinate[2], coordinate[4], coordinate[6])
    y_min = min(coordinate[1], coordinate[3], coordinate[5], coordinate[7])
    x_max = max(coordinate[0], coordinate[2], coordinate[4], coordinate[6])
    y_max = max(coordinate[1], coordinate[3], coordinate[5], coordinate[7])
    return x_min, x_max, y_min, y_max

# 裁剪函数，返回裁剪后的 img_list
def pt2img(strlist, img):
    # cv2.imshow("img", img)
    imglist = []
    for line in strlist:
        # print(line)
        x_min, x_max, y_min, y_max = calcu_crop_para(line)
        img_crop = img[y_min:y_max, x_min:x_max]
        # print(type(img_crop))
        # cv2.imshow("crop", img_crop)
        # img_crop = cv2.resize(img_crop, (280, 32))
        # img_crop = cv2.resize(img_crop, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        imglist.append(img_crop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    # while (1):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    return imglist

# # 批量读取函数，读取完整的图片 和 坐标信息  注意： image 和 txt 名字必须相同
# def read_img_txt(imgfile_path, txtfile_path):
#     images = []   # 存放目录下所有图片
#     bbox = []  # 存放所有txt文件的坐标信息，其顺序和 images 中存放图片的顺序一致（一一对应）
#     box_str = []  # 存放一个txt的坐标信息，type = str
#     box_int = []  # 存放一个txt的坐标信息，type = int
#     # 读取目录下所有的 images
#     for file in os.listdir(imgfile_path):
#         img_path = os.path.join(imgfile_path, file)  # 每一张图片的路径
#         img = cv2.imread(img_path)  # 读取图片
#         images.append(img)  # 将图片存放到 images 列表中
#     # 读取目录下所有的 txt
#     for txt in os.listdir(txtfile_path):
#         txt_path = os.path.join(txtfile_path, txt)  # 每一个txt的路径
#         with open(txt_path, 'r') as f:
#             lines = f.read().splitlines()  # 读取txt中的每行，去除换行符
#         for line in lines:
#             box_str.append(line.split(','))   # 将坐标信息存放到 box_str 中
#         # 将坐标列表转化为int类型
#         for box in box_str:
#             box = list(map(int, box))
#             box_int.append(box)  # 将坐标信息转为 int型 并存放到 box_int 中
#         bbox.append(box_int)  # 将坐标信息存放到 bbox 中
#         # 清空操作
#         box_str = []
#         box_int = []
#     return images, bbox

# 单张图片和bbox读取函数，读取完整的图片 和 坐标信息  注意： image 和 txt 名字必须相同
def read_img_txt(imgfile_path, txtfile_path):
    # images = []   # 存放目录下所有图片
    bbox = []  # 存放所有txt文件的坐标信息，其顺序和 images 中存放图片的顺序一致（一一对应）
    box_str = []  # 存放一个txt的坐标信息，type = str
    box_int = []  # 存放一个txt的坐标信息，type = int
    # 读取单张images
    img = cv2.imread(imgfile_path)  # 读取图片
    img_name = (os.path.basename(imgfile_path)).split('.')[0]
    # images.append(img)  # 将图片存放到 images 列表中
    # 读取目录下和图片名字相同的 txt
    for txt in os.listdir(txtfile_path):
        txt_path = os.path.join(txtfile_path, txt)  # 每一个txt的路径
        txt_name = txt.split('.')[0]
        if txt_name == img_name:
            with open(txt_path, 'r') as f:
                lines = f.read().splitlines()  # 读取txt中的每行，去除换行符
            for line in lines:
                box_str.append(line.split(','))   # 将坐标信息存放到 box_str 中
            # 将坐标列表转化为int类型
            for box in box_str:
                box = list(map(int, box))
                box_int.append(box)  # 将坐标信息转为 int型 并存放到 box_int 中
            bbox.append(box_int)  # 将坐标信息存放到 bbox 中
    return img, bbox

# # 以下为测试代码
# im_path = 'C:/Users/soak/Desktop/match_ctpn_ocr/images/'  # 图片的路径
# tx_path = 'C:/Users/soak/Desktop/match_ctpn_ocr/txt/'  # txt的路径
#
# bbox_str = []
# bbox = []   #
# image = []  # 存放所有图片
#
# # 读取txt中检测框的坐标
# for tx in os.listdir(tx_path):
#     txPath = os.path.join(tx_path, tx)
#     with open(txPath, 'r') as f:
#         lines = f.read().splitlines()
#     for line in lines:
#         bbox_str.append(line.split(','))
#
# # 将坐标列表转化为int类型
# for box in bbox_str:
#     box = list(map(int, box))
#     bbox.append(box)
# print(bbox)
# print(len(bbox))
#
# # 读取图片
# for im in os.listdir(im_path):
#     imgPath = os.path.join(im_path, im)
#     img = cv2.imread(imgPath)
#     crop_list = pt2img(bbox, img)
#     # print(crop_list)
#     for i in range(len(crop_list)):
#         crop_list[i] = cv2.resize(crop_list[i],(560, 60))
#     for i in crop_list:
#         cv2.imshow("crop", i)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

