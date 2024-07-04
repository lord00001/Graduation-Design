import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import match
import CTPN.ctpn
from 基于形态学操作法的文字区域检测 import Mm
from layout_analysis.layout_analysis import *
from layout_analysis.generate_excel import *
from data_preprocess.data_preprocess import *
from RedChapter_detec.RedChapter_detec import *
import random

# 手写字识别部分
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
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
def AHR(cnn_model, img):
    cnn_model.eval()
    model = cnn_model
    AH_label = cls_inference(model, img)
    if AH_label == 0:
        print('签名')
    if AH_label == 1:
        print('合格')
    return AH_label

# 手写字 / 印刷字
def HPR(cnn_model, img):
    cnn_model.eval()
    model = cnn_model
    HP_label = cls_inference(model, img)
    if HP_label == 0:
        print('手写')
    if HP_label == 1:
        print('印刷')
    return HP_label


def parse_arg(file_path):
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default=file_path, help='待识别文字图片保存的位置')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoints/mixed_second_finetune_acc_97P7.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

# 对图片进行识别  输出：识别的文字
def recognition(config, img, model, converter, device):

    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.W / w, fy=config.MODEL.IMAGE_SIZE.H / h,
    #                  interpolation=cv2.INTER_CUBIC)
    #
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W, 1))

    # second step: keep the ratio of image's text same with training
    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    print('results: {0}'.format(sim_pred))
    return sim_pred


# # 添加连接代码  input： 完整的图片，坐标txt，一次只能放一张图和一个txt文件
# # output： 将识别的文字保存到txt文件里
# if __name__== '__main__':
#     imgfile_path = './test/images/'
#     txtfile_path = './test/txt/'
#     save_path = './test/OCR_result/'
#     ctpn_model_path = './CTPN/model_save/ctpn_99.pth'
#     ctpn_base_model = 'shufflenet_v2_x1_0'
#     ctpn_detect_type = 'H'  # 'O' or 'H' 检测格式 ‘倾斜’，‘垂直’
#     result = []  # 存放所有图片的识别结果
#     sigel_result = []  # 存放单张图的识别结果
#     images = []  # 存放所有待识别的图片
#     bbox = []  # 存放所有的坐标信息
#     crop_img_list = []  # 存放所有裁剪信息
#     # 文字区域定位
#     CTPN.ctpn.txt_area_pos(imgfile_path, txtfile_path, ctpn_model_path, ctpn_base_model, ctpn_detect_type)
#     # 读取所有的图片和坐标信息
#     images, bbox = match.read_img_txt(imgfile_path, txtfile_path)
#     # 根据坐标信息进行裁剪
#     for i in range(len(images)):  # len(images) = len(bbox)
#         crop_img_list.append(match.pt2img(bbox[i], images[i]))
#     # 进行识别
#     for file in os.listdir(imgfile_path):
#         file_path = os.path.join(imgfile_path, file)
#         config, args = parse_arg(file_path)
#         device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#
#         model = crnn.get_crnn(config).to(device)
#         # print('loading pretrained model from {0}'.format(args.checkpoint))
#         checkpoint = torch.load(args.checkpoint)
#         if 'state_dict' in checkpoint.keys():
#             model.load_state_dict(checkpoint['state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
#
#     # 开始识别
#     started = time.time()
#     for sigle_crop_img in crop_img_list:  # crop_img_list 里存放的是目录里所有的图片的裁剪信息
#         for crop_img in sigle_crop_img:   # sigle_crop_img 里存放的是一张图片的裁剪信息
#             img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#             converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
#             sigel_result.append(recognition(config, img, model, converter, device))
#             # print('elapsed time: {0}'.format(finished - started))
#         result.append(sigel_result)
#         sigel_result = []
#     finished = time.time()
#     # 识别结束
#     # 将识别结果存放到txt中
#     for i in range(len(result)):
#         with open('./test/OCR_result/报告识别结果_{}.txt'.format(i), 'a+', encoding='utf-8') as f:
#             for data in result[i]:
#                 f.write(data + '\n')
#         f.close()

# 添加连接代码  input： 完整的图片，坐标txt，一次只能放一张图和一个txt文件
# output： 将识别的文字保存到txt文件里
# 识别单张图片
def OCR(imgfile_path, detect_type):
    # imgfile_path = './test/images/test1.jpg'
    RedC_Hand_information = []  # 手写字红章检测信息
    # 读取图片， 仅能读取单张图片
    img = cv2.imread(imgfile_path)
    # 对图片进行倾斜矫正
    # img_tilted = tilt_correction(img)
    img_tilted = img
    # 对图片进行预处理
    img_pre = data_preprocess(img)
    img_name = (os.path.basename(imgfile_path)).split('.')[0]
    img_pre_path = './test/images_preprocess/' + img_name + '.jpg'
    cv2.imwrite(img_pre_path, img_pre)
    AH_model_path = './HCR/model/HAR_cnn_model.pt'
    HP_model_path = './HCR/model/HPR_cnn_model.pt'
    txtfile_path = './test/txt/'
    detect_images_path = './test/detect_images/'
    save_path = './test/OCR_result/'
    ctpn_model_path = './CTPN/model_save/ctpn_99.pth'
    # ctpn_model_path = './CTPN/model_save/ctpn_83.pth'
    ctpn_base_model = 'shufflenet_v2_x1_0'
    ctpn_detect_type = 'H'  # 'O' or 'H' 检测格式 ‘倾斜’，‘垂直’
    sigel_result = []  # 存放单张图的识别结果
    # bbox = []  # 存放所有的坐标信息
    # crop_img_list = []  # 存放所有裁剪信息
    if detect_type == 1 or detect_type == 3 or detect_type == 4:   # 采用 形态学操作法 进行文字区域定位
        Mm(img_tilted, detect_images_path, imgfile_path)
    elif detect_type == 2:  # 采用 CTPN 网络进行文字区域定位
        CTPN.ctpn.txt_area_pos(imgfile_path, txtfile_path, ctpn_model_path, ctpn_base_model, ctpn_detect_type)
    # elif detect_type == 3: # 采用版面分析进行文字区域定位
    # 读取所有的图片和坐标信息
    img, bbox = match.read_img_txt(imgfile_path, txtfile_path)
    # 根据坐标信息进行裁剪
    # for i in range(len(images)):  # len(images) = len(bbox)
    # for i in range(len(bbox[0])):
    crop_img_list = match.pt2img(bbox[0], img)

    # 进行识别
    config, args = parse_arg(imgfile_path)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    # print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # 开始识别
    # 文字框坐标信息
    word_coor = bbox[0]
    temp = 0
    started = time.time()
    # for sigle_crop_img in crop_img_list:  # crop_img_list 里存放的是目录里所有的图片的裁剪信息
    print('bbox: ', bbox)
    print('len(bbox): ', len(bbox[0]))
    print('len(crop_img_list): ', len(crop_img_list))
    # 检测红章
    RedC_Hand_information.append(RedChapter_detec(img_tilted))
    # 加载手写字识别部分的模型
    AH_model = torch.load(AH_model_path)
    HP_model = torch.load(HP_model_path)

    for crop_img in crop_img_list:   # sigle_crop_img 里存放的是一张图片的裁剪信息
        # 手写字检测
        if HPR(HP_model, crop_img) == 0:  # 如果检测到手写字
            label = AHR(AH_model, crop_img)
            if label == 0:  # 检测到“签名”
                RedC_Hand_information.append('检测到手写字“签名”')
                cv2.imwrite('./test/RedChapter_handwriting_output/autograph{}.jpg'.format(random.random()), crop_img)
            if label == 1:  # 检测到“合格”
                RedC_Hand_information.append('检测到手写字“合格”')
                cv2.imwrite('./test/RedChapter_handwriting_output/handwriting{}.jpg'.format(random.random()), crop_img)
        img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        sigel_result.append(recognition(config, img, model, converter, device))

        word_coor[temp].append(recognition(config, img, model, converter, device))
        temp += 1

        # print('elapsed time: {0}'.format(finished - started))
    finished = time.time()
    # 识别结束
    print('word_coor: ', word_coor)

    information = []
    for i in RedC_Hand_information:
        if i not in information:
            information.append(i)

    if detect_type != 3 and detect_type != 4:          # 3是无表格识别
        # 版面分析
        frame = layout_analysis_to_frame(img_pre_path)
        table_frame = generate_table_frame(frame, word_coor)
        # table_frame = [[[200, 98, 10, '基本信患']], [[239, 98, 2, '检验报告编号'], [240, 326, 2], [245, 593, 1], [249, 785, 5]],
        #                [[277, 99, 2, '号牌号码'], [277, 326, 2, '苏BG6IAC'], [282, 593, 1, '所有人'], [286, 785, 5]],
        #                [[330, 100, 2, '车辆类型'], [331, 327, 2, '轻型栏板货车'], [334, 593, 1, '品牌/型"'], [337, 785, 5]],
        #                [[366, 101, 2], [367, 327, 2], [371, 593, 1, '道路运龄证号', '品牌/型"'], [373, 785, 5]],
        #                [[405, 101, 2, '注册登记H期', '使用性质'], [407, 327, 2, '20I4-0-I0'], [410, 593, 1, 'HI厂H期', '道路运龄证号'],
        #                 [411, 784, 2, '薛鹤'], [413, 1055, 3, '检验日期']],
        #                [[441, 100, 2, '藏解'], [443, 327, 2], [446, 593, 1, 'HI厂H期'], [447, 784, 2, '薛鹤'], [449, 1055, 3]],
        #                [[504, 99, 2, '检验类别'], [507, 327, 3], [509, 784, 2], [511, 1054, 3]], [[555, 98, 2], [558, 327, 8]],
        #                [[592, 98, 8], [598, 1131, 1], [598, 1257, 1]],
        #                [[630, 97, 2], [632, 326, 3], [634, 783, 2], [636, 1054, 1], [636, 1131, 1], [636, 1257, 1]],
        #                [[697, 98, 2], [699, 326, 3], [701, 783, 2], [702, 1054, 2], [702, 1361, 1]], [[734, 98, 10]],
        #                [[773, 98, 1, '序号'], [773, 238, 3], [775, 625, 1, '结果判定'], [775, 782, 3], [776, 1252, 1, '朱己宇']],
        #                [[810, 98, 1], [811, 238, 3], [812, 624, 1, '合格'], [812, 782, 3], [813, 1252, 1, '朱入宇', '朱己宇']],
        #                [[848, 98, 1], [849, 237, 3], [850, 624, 1, '合格', '合格'], [850, 781, 3],
        #                 [851, 1252, 1, '朱飞宇', '朱入宇']],
        #                [[885, 98, 1], [885, 237, 3], [886, 623, 1, '合格'], [887, 781, 3], [888, 1252, 1, '朱飞宇', '朱飞宇']],
        #                [[923, 98, 1], [924, 236, 3], [925, 623, 1, '合格', '合格'], [925, 781, 3],
        #                 [926, 1252, 1, '朱入宇', '朱飞宇']],
        #                [[960, 98, 1], [960, 236, 3], [961, 622, 1, '合格'], [962, 780, 3], [963, 1252, 1, '朱入宇']],
        #                [[998, 99, 1], [999, 235, 3], [1000, 622, 1, '合格'], [1000, 780, 3], [1002, 1252, 1]],
        #                [[1035, 98, 1], [1036, 235, 3], [1036, 622, 1, '合格'], [1037, 780, 3], [1039, 1252, 1]],
        #                [[1074, 98, 1], [1074, 235, 3], [1075, 622, 1], [1075, 780, 3], [1077, 1252, 1]],
        #                [[1110, 98, 1, '件号'], [1110, 234, 3], [1111, 621, 1, '检验结果'], [1112, 779, 2],
        #                 [1113, 1052, 1, '合格/合格', '结果判定'], [1113, 1251, 1]],
        #                [[1148, 98, 1], [1149, 234, 3, '一轴空载削动率(8)/不平衡廖(%)'], [1150, 621, 1, '81.4/20.5'], [1151, 779, 2],
        #                 [1152, 1052, 1, '合格/合格', '合格/合格'], [1153, 1251, 1]],
        #                [[1185, 98, 1], [1185, 234, 3, '二轴空载帕动率(M/不件衡率(%'], [1186, 621, 1, '68.6/8.0'], [1187, 779, 2],
        #                 [1189, 1052, 1, '合格/合格', '合格/合格'], [1189, 1250, 1]],
        #                [[1187, 97, 1], [1185, 234, 3, '二轴空载帕动率(M/不件衡率(%'], [1186, 621, 1, '68.6/8.0'], [1187, 779, 2],
        #                 [1189, 1052, 1, '合格/合格', '合格/合格'], [1189, 1250, 1]],
        #                [[1224, 97, 1], [1224, 234, 3, '整车制动率(%/驻车削动率()'], [1226, 621, 1, '76.I/26.5'], [1227, 779, 2],
        #                 [1228, 1052, 1, '合格/合格'], [1229, 1250, 1]],
        #                [[1261, 97, 1], [1261, 234, 3, '住照灯左外灯远光发光强度(ed'], [1263, 621, 1, '15100', '76.I/26.5'],
        #                 [1263, 779, 2], [1265, 1052, 1], [1266, 1250, 1]],
        #                [[1299, 96, 1], [1300, 234, 3, '前照灯右外灯远光发光强度(ab_'], [1301, 621, 1, 'I5100'], [1302, 779, 2],
        #                 [1303, 1051, 1], [1304, 1250, 1]],
        #                [[1336, 96, 1], [1337, 234, 3], [1338, 621, 1], [1339, 779, 2], [1340, 1051, 1], [1341, 1249, 1]],
        #                [[1375, 96, 1], [1375, 234, 3], [1376, 621, 1], [1377, 778, 2], [1378, 1051, 1], [1380, 1249, 1]],
        #                [[1411, 96, 1], [1411, 234, 3], [1413, 621, 1], [1414, 778, 2], [1415, 1051, 1], [1416, 1249, 1]],
        #                [[1450, 96, 1], [1450, 234, 3], [1451, 621, 1], [1452, 778, 2], [1453, 1051, 1], [1455, 1249, 1]],
        #                [[1487, 96, 1], [1487, 234, 3], [1488, 621, 1], [1489, 778, 2], [1490, 1050, 1, 'I六、二维条码'],
        #                 [1492, 1248, 1]],
        #                [[1525, 97, 1, 'I五、型议'], [1526, 234, 3], [1527, 621, 1], [1527, 778, 2], [1529, 1050, 1, 'I六、二维条码'],
        #                 [1530, 1248, 1]], [[1562, 97, 7], [1566, 1050, 3]],
        #                [[1765, 98, 1], [1766, 233, 6], [1767, 1048, 3]], [[1881, 100, 1], [1882, 233, 9]]]

        print("table_frame-last", table_frame)
        generate_table(table_frame, information)

    # 将识别结果存放到txt中
    with open('./test/OCR_result/报告{}的识别结果.txt'.format(img_name), 'a+', encoding='utf-8') as f:
        for data in sigel_result:
            f.write(data + '\n')
    f.close()

    # 保存识别文字+坐标信息
    with open('./test/OCR_result/报告{}的文本坐标信息.txt'.format(img_name), 'a+', encoding='utf-8') as f:
        for line in word_coor:
            for word in line:
                if word == line[-1]:
                    f.write(word + '\n')
                else:
                    f.write(str(word) + ',')
    f.close()

    print("识别结果：", sigel_result)
    return sigel_result

# OCR('./test/images/test1.jpg', 'H')