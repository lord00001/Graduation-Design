# -*- coding: utf-8 -*-
import os

label = []
image = []
#设定文件路径
path_image = './data/images/'
# path_label = './data/ICPR_text_train_part2_20180313/label'

i = 1
j = 1
# #对目录下的文件进行遍历
# for file in os.listdir(path_label):
#     label.append(file)
#     new_name = file.replace(file, "{}.txt".format(i))
#     os.rename(os.path.join(path_label, file), os.path.join(path_label, new_name))
#     i += 1

for file in os.listdir(path_image):
    image.append(file)
    new_name = file.replace(file, "{}.jpg".format(j))
    os.rename(os.path.join(path_image, file), os.path.join(path_image, new_name))
    j += 1

# print(label)
print(image)
# #判断是否是文件
#     if os.path.isfile(os.path.join(path,file))==True:
# #设置新文件名
#         new_name=file.replace(file,"{}.txt".format(i))
# #重命名
#         os.rename(os.path.join(path,file),os.path.join(path,new_name))
#         i+=1
# #结束
# print ("End")
