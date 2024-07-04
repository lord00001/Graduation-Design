# -*- coding: utf-8 -*-

"""
******不改变原始xml的一些数据增强方法 type 1-10*******
把增强后的图像和xml一起放入新的文件夹
rootpath:picture_xml原始路径
savepath：picture_xml保存路径
*******改变原始 xml的一些数据增强方法  type 11-15******
修改图片的同时修改对应的xml
file_path:传入类别的信息txt，最好和生成labelmap的顺序一致
rootpath:picture_xml原始路径
savepath：picture_xml保存路径

11:自定义裁剪，图像大小 w,h，例如 w=400,h=600
12：自定义平移，平移比例 w,h [0-1] 例如w=0.1,h=0,2
13：自定义缩放，调整图像大小 w,h,例如 w=400,h=600
14：图像翻转
15:图像任意旋转，传入旋转角度列表anglelist=[90,-90]

"""
import cv2
import random
import math
import os, shutil
import numpy as np
from PIL import Image, ImageStat
from skimage import exposure
import matplotlib.pyplot as plt
import tensorlayer as tl
from scipy import ndimage
from lxml.etree import Element, SubElement, tostring
import xml.etree.ElementTree as ET


def hisColor_Img(path):
    """
    对图像直方图均衡化
    :param path: 图片路径
    :return: 直方图均衡化后的图像
    """
    img = cv2.imread(path)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])  # equalizeHist(in,out)
    cv2.merge(channels, ycrcb)
    img_eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
    return img_eq


def clahe_Img(path, ksize):
    """
    :param path: 图像路径
    :param ksize: 用于直方图均衡化的网格大小，默认为8
    :return: clahe之后的图像
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ksize, ksize))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image = cv2.merge([b, g, r])
    return image


def whiteBalance_Img(img):
    """
    对图像白平衡处理
    """
    # img = cv2.imread(path)
    b, g, r = cv2.split(img)
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cr = 0.5 * r - 0.419 * g - 0.081 * b
    Cb = -0.169 * r - 0.331 * g + 0.5 * b
    Mr = np.mean(Cr)
    Mb = np.mean(Cb)
    Dr = np.var(Cr)
    Db = np.var(Cb)
    temp_arry = (np.abs(Cb - (Mb + Db * np.sign(Mb))) < 1.5 * Db) & (
            np.abs(Cr - (1.5 * Mr + Dr * np.sign(Mr))) < 1.5 * Dr)
    RL = Y * temp_arry
    # 选取候选白点数的最亮10%确定为最终白点，并选择其前10%中的最小亮度值
    L_list = list(np.reshape(RL, (RL.shape[0] * RL.shape[1],)).astype(np.int))
    hist_list = np.zeros(256)
    min_val = 0
    sum = 0
    for val in L_list:
        hist_list[val] += 1
    for l_val in range(255, 0, -1):
        sum += hist_list[l_val]
        if sum >= len(L_list) * 0.1:
            min_val = l_val
            break
    # 取最亮的前10%为最终的白点
    white_index = RL < min_val
    RL[white_index] = 0
    # 计算选取为白点的每个通道的增益
    b[white_index] = 0
    g[white_index] = 0
    r[white_index] = 0
    Y_max = np.max(RL)
    b_gain = Y_max / (np.sum(b) / np.sum(b > 0))
    g_gain = Y_max / (np.sum(g) / np.sum(g > 0))
    r_gain = Y_max / (np.sum(r) / np.sum(r > 0))
    b, g, r = cv2.split(img)
    b = b * b_gain
    g = g * g_gain
    r = r * r_gain
    # 溢出处理
    b[b > 255] = 255
    g[g > 255] = 255
    r[r > 255] = 255
    res_img = cv2.merge((b, g, r))
    return res_img


def bright_Img(path, ga, flag):
    """
    亮度增强 Tensorlayer
    :param ga: ga为gamma值，>1亮度变暗，<1亮度变亮
    :param flag:True: 亮度值为(1-ga,1+ga)
                False:亮度值为ga,默认为1
    :return: 亮度增强后的图像
    """
    image = tl.vis.read_image(path)
    tenl_img = tl.prepro.brightness(image, gamma=ga, is_random=flag)
    return tenl_img


def illumination_Img(path, ga, co, sa, flag):
    """
    亮度,饱和度，对比度增强 Tensorlayer
    :param ga: ga为gamma值，>1亮度变暗，<1亮度变亮
    :param co: 对比度值，1为原始值
    :param sa: 饱和度值，1为原始值
    :param flag:True: 亮度值为(1-ga,1+ga)，对比度(1-co,1+co)，饱和度(1-sa,1+sa)
                False:亮度值为ga,对比度co,饱和度sa
    :return:增强后的结果
    """
    image = tl.vis.read_image(path)
    tenl_img = tl.prepro.illumination(image, gamma=ga, contrast=co, saturation=sa, is_random=flag)
    return tenl_img


def create_mask(imgpath):
    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return mask


def xiufu_Img(imgpath, maskpath):
    """
    去除图像上的高光部分
    """
    src_ = cv2.imread(imgpath)
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    # 缩放因子(fx,fy)
    res_ = cv2.resize(src_, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_CUBIC)
    dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)
    return dst


def image_brightness(rgb_image):
    '''
    检测图像亮度(基于RMS)
    '''
    stat = ImageStat.Stat(rgb_image)
    r, g, b = stat.rms
    return math.sqrt(0.241 * (r ** 2) + 0.691 * (g ** 2) + 0.068 * (b ** 2))


def calc_gamma(brightness):
    return brightness / 127.0


def image_gamma_transform(pil_im, gamma):
    image_arr = np.array(pil_im)
    image_arr2 = exposure.adjust_gamma(image_arr, gamma)
    if len(image_arr.shape) == 3:  # 格式为(height(rows), weight(colums), 3)
        r = Image.fromarray(np.uint8(image_arr[:, :, 0]))
        g = Image.fromarray(np.uint8(image_arr[:, :, 1]))
        b = Image.fromarray(np.uint8(image_arr[:, :, 2]))
        image = Image.merge("RGB", (r, g, b))
        return image
    elif len(image_arr.shape) == 2:  # 格式为(height(rows), weight(colums))
        return Image.fromarray(np.uint8(image_arr))


def autobright_Img(rootpath, savepath):
    """
    自适应亮度增强
    """
    list = os.listdir(rootpath)
    for i in range(0, len(list)):
        path = os.path.join(rootpath, list[i])
        if os.path.isfile(path):
            if list[i].endswith("jpg") or list[i].endswith("JPG") or list[i].endswith("png") or list[i].endswith("PNG"):
                print("adjust_bright running....")
                print(list[i])
                im = Image.open(path)
                brightness = image_brightness(im)
                newimage = np.array(image_gamma_transform(im, calc_gamma(brightness)))
                newname = "adjust_bright" + list[i][:-4]
                saveflie = os.path.join(savepath, newname + ".jpg")
                plt.imsave(saveflie, newimage)
                shutil.copyfile(os.path.join(rootpath, list[i][:-4] + ".xml"),
                                os.path.join(savepath, newname + ".xml"))


def probability_random_event(rate, event):
    """随机变量的概率函数"""
    #
    # 参数rate为list<int>
    # 返回概率事件的下标索引
    start = 0
    index = 0
    randnum = random.randint(1, sum(rate))

    for index, scope in enumerate(rate):
        start += scope
        if randnum <= start:
            break
    return event[index]


def erase_Img(root_path, save_path):
    """
    随机遮挡
    """
    for file in os.listdir(root_path):
        file_name, extension = os.path.splitext(file)
        if extension == '.xml':
            print("erase running....")
            print(file_name + ".jpg")
            xml_path = os.path.join(root_path, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_path = os.path.join(root_path, file_name + '.jpg')
            image = cv2.imread(image_path)
            for obj in root.findall('object'):
                # 文件夹图像遮挡的比例，参数可修改
                is_erase = probability_random_event([7, 3], [True, False])
                if is_erase:
                    # boundingbox遮挡的方向，参数可修改
                    erase_orientation = probability_random_event([6, 2, 1, 1],
                                                                 ['down', 'up', 'left', 'right'])
                    # 遮挡方块的大小，参数可修改
                    erase_scope = random.uniform(0.1, 0.3)
                    xml_box = obj.find('bndbox')
                    _xmin = int(xml_box.find('xmin').text)
                    _xmax = int(xml_box.find('xmax').text)
                    _ymin = int(xml_box.find('ymin').text)
                    _ymax = int(xml_box.find('ymax').text)
                    box_width = _xmax - _xmin
                    box_height = _ymax - _ymin
                    new_xmin, new_xmax, new_ymin, new_ymax = _xmin, _xmax, _ymin, _ymax
                    if erase_orientation == 'down':
                        new_ymax = int(_ymax - box_height * erase_scope)
                        image[new_ymax:_ymax, new_xmin:new_xmax, :] = 255
                    if erase_orientation == 'up':
                        new_ymin = int(_ymin + box_height * erase_scope)
                        image[_ymin:new_ymin, new_xmin:new_xmax, :] = 255
                    if erase_orientation == 'left':
                        new_xmin = int(_xmin + box_width * erase_scope)
                        image[new_ymin:new_ymax, _xmin:new_xmin, :] = 255
                    if erase_orientation == 'right':
                        new_xmax = int(_xmax - box_width * erase_scope)
                        image[new_ymin:new_ymax, new_xmax:_xmax, :] = 255
                    cv2.imwrite(os.path.join(save_path, "earse_" + file_name + '.jpg'), image)
                    xml_box.find('xmin').text = str(new_xmin)
                    xml_box.find('ymin').text = str(new_ymin)
                    xml_box.find('xmax').text = str(new_xmax)
                    xml_box.find('ymax').text = str(new_ymax)
                    tree.write(os.path.join(save_path, "earse_" + file_name + '.xml'))


def blur_Img(rootpath, savepath, ksize, new_rate):
    """
    随机模糊图像
    """
    img_list = []
    for imgfiles in os.listdir(rootpath):
        if (imgfiles.endswith("jpg") or imgfiles.endswith("JPG")):
            img_list.append(imgfiles)
    filenumber = len(img_list)
    rate = new_rate  # 自定义抽取文件夹中图片的比例，参数可修改
    picknumber = int(filenumber * rate)
    sample = random.sample(img_list, picknumber)
    for name in sample:
        print("blur running....")
        print(name)
        namepath = os.path.join(rootpath, name)
        ori_img = cv2.imread(namepath)
        size = random.choice(ksize)  # 设置高斯核的大小，参数可修改，size>9，小于9图像没有变化
        kernel_size = (size, size)
        image = cv2.GaussianBlur(ori_img, ksize=kernel_size, sigmaX=0, sigmaY=0)
        cv2.imwrite(os.path.join(savepath, "blur_" + name), image)
        shutil.copyfile(os.path.join(rootpath, name.split(".")[0] + ".xml"),
                        os.path.join(savepath, "blur_" + name.split(".")[0] + ".xml"))


def compress_Img(infile_path, outfile_path, pic_size):
    """
    压缩图像
    不改变图片尺寸压缩到指定大小
    :param infile: 压缩源文件
    :param outfile: 压缩文件保存地址
    """
    count = 0
    for infile in os.listdir(infile_path):
        if infile.endswith(".jpg") or infile.endswith(".GPG") or \
                infile.endswith(".jpeg") or infile.endswith(".GPEG"):
            print("compress_ running....")
            print(infile)
            filename, extend_name = os.path.splitext(infile)
            img_path = os.path.join(infile_path, infile)
            imgsaved_path = os.path.join(outfile_path, "compress_" + infile)
            img = cv2.imread(img_path, 1)
            # 获取文件大小:KB
            img_size = os.path.getsize(img_path) / 1024
            if img_size > pic_size:
                cv2.imwrite(imgsaved_path, img, [cv2.IMWRITE_JPEG_QUALITY, 30])
                shutil.copyfile(os.path.join(infile_path, filename + ".xml"),
                                os.path.join(outfile_path, "compress_" + filename + ".xml"))
            else:
                shutil.copyfile(img_path, imgsaved_path)
                shutil.copyfile(os.path.join(infile_path, filename + ".xml"),
                                os.path.join(outfile_path, "compress_" + filename + ".xml"))





def localStd(img):
    # 归一化
    # img = img / 255.0
    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img, (21, 21))
    reslut_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    reslut_2 = cv2.blur(img_2, (21, 21))

    reslut = np.sqrt(np.maximum(reslut_2 - reslut_1, 0))
    return reslut


def get_reflect(img, img_illumination):
    # get_img_illumination = get_illumination(img)
    get_img_reflect = (img + 0.001) / (img_illumination + 0.001)
    return get_img_reflect


def enhancement_reflect(img):
    # 通过高斯滤波器
    gaussian_blur_img = cv2.GaussianBlur(img, (21, 21), 0)
    enhancement_reflect_img = img * gaussian_blur_img
    return enhancement_reflect_img


def get_enhancment_img(img_enhance_illumination, img_enahnce_reflect):
    img = img_enhance_illumination * img_enahnce_reflect
    img = img.astype('uint8')
    return img


def read_img_from_disk(file_path):
    # 0. 读取图像
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    return img


def get_illumination(img):
    return cv2.GaussianBlur(img, (15, 15), 0)


"""
enhancment_illumination 增强反射分量，传入反射分量，返回增强后的反射分量
"""


def enhancment_illumination(img_illumination):
    img_hsv = cv2.cvtColor(img_illumination, cv2.COLOR_BGR2HSV)
    img_hsv = (img_hsv - np.min(img_hsv)) / (np.max(img_hsv) - np.min(img_hsv))
    h, s, v = cv2.split(img_hsv)
    wsd = 5
    gm = np.mean(v) / (1 + wsd * np.std(v)) # 一个数字
    cst = localStd(v)   # 300 * 400 的矩阵
    lm = gm * v /(1 + wsd * cst)    # 300 * 400 的矩阵
    c = np.exp(gm)      # 一个常数
    wg = v ** 0.2       # 300 *400
    wl = 1- wg
    outM = v**c / (v**c +(wl * lm)**c + (wg * gm)**c + 0.001)
    outM = 1.5 * outM - 0.5 * cv2.GaussianBlur(outM, (21, 21), 0)
    outM = (outM - np.min(outM))/(np.max(outM) - np.min(outM))
    paramerter = 0.9
    img_illumination[:, :, 0] = outM * (img_illumination[:, :, 0] / (v + 0.01))**paramerter
    img_illumination[:, :, 1] = outM * (img_illumination[:, :, 1] / (v + 0.01))**paramerter
    img_illumination[:, :, 2] = outM * (img_illumination[:, :, 2] / (v + 0.01))**paramerter
    return img_illumination


def data_preprocess(img):
    # img = cv2.imread(rootpath + '/%s' % filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 霍夫变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
    if x1 == x2 or y1 == y2:
        img = img
    else:
        t = float(y2 - y1) / (x2 - x1)
        rotate_angle = math.degrees(math.atan(t))
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
        img = ndimage.rotate(img, rotate_angle)
        # cv2.imwrite(savepath + '/%s' % filename, rotate_img)

    # 去除高光
    img_illumination = get_illumination(img)  # 获得高频分量
    img_reflect = get_reflect(img, img_illumination)  # 获得反射分量
    img_enhancement_reflect = enhancement_reflect(img_reflect)  # 增强反射分量
    img_enhancement_illumination = enhancment_illumination(img_illumination)  # 增强照射分量
    img_done = get_enhancment_img(img_enhancement_illumination, img_reflect)  # 照射分量与反射分量融合
    # 白平衡
    enhance3_img = whiteBalance_Img(img_done)
    return enhance3_img



# if __name__ == '__main__':
#     rootpath = "./images/"
#     savepath = "./output/"
#     # type 6 去除高光需要给mask蒙版的保存地址
#     masksavepath = "./date_set/mask"
#     file_path = "./data_set/wcp.txt"
#     # 1.直方图均衡化 2.clahe自适应对比度直方图均衡化 3.白平衡 4.亮度增强 5.亮度，饱和度，对比度增强 6.去除图像上的高光部分
#     imgfiles = os.listdir(rootpath)
#     for filename in os.listdir(rootpath):
#
#         img = cv2.imread(rootpath + '/%s' % filename)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, 50, 150, apertureSize=3)
#
#         # 霍夫变换
#         lines = cv2.HoughLines(edges, 1, np.pi / 180, 0)
#         for rho, theta in lines[0]:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             x1 = int(x0 + 1000 * (-b))
#             y1 = int(y0 + 1000 * (a))
#             x2 = int(x0 - 1000 * (-b))
#             y2 = int(y0 - 1000 * (a))
#         if x1 == x2 or y1 == y2:
#             continue
#         t = float(y2 - y1) / (x2 - x1)
#         rotate_angle = math.degrees(math.atan(t))
#         if rotate_angle > 45:
#             rotate_angle = -90 + rotate_angle
#         elif rotate_angle < -45:
#             rotate_angle = 90 + rotate_angle
#         rotate_img = ndimage.rotate(img, rotate_angle)
#         cv2.imwrite(savepath + '/%s' % filename, rotate_img)
#
#     print(imgfiles)
#     for i in range(0, len(imgfiles)):
#         path = os.path.join(savepath, imgfiles[i])
#         print(imgfiles[i])
#         print(path)
#         if os.path.isfile(path):
#             if (imgfiles[i].endswith("jpg") or imgfiles[i].endswith("JPG")):
#                 # 去除高光
#                 img = read_img_from_disk(path)  # 读取图像
#                 img_illumination = get_illumination(img)  # 获得高频分量
#                 img_reflect = get_reflect(img, img_illumination)  # 获得反射分量
#                 img_enhancement_reflect = enhancement_reflect(img_reflect)  # 增强反射分量
#                 img_enhancement_illumination = enhancment_illumination(img_illumination)  # 增强照射分量
#                 img_done = get_enhancment_img(img_enhancement_illumination, img_reflect)  # 照射分量与反射分量融合
#                 # # 结果
#                 cv2.imwrite(path, img_done)
#                 # 白平衡
#                 print("whiteBalance running....")
#                 enhance3_img = whiteBalance_Img(path)
#                 cv2.imwrite(path, enhance3_img)

