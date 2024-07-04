# coding:utf-8
import os
import xlwt

# tabel_frame = [
#        [[1, 1, "文字1"]],
#        [[1, 1, "文字5"], [1, 1, "文字6"], [1, 1, "文字7"], [1, 1, "文字8"]],
#        [[1, 1, "文字9"], [1, 1, "文字10"], [1, 1, "文字11"], [1, 1, "文字12"]],
#        [[1, 1, "文字13"], [1, 1, "文字14"], [1, 1, "文字15"], [1, 1, "文字16"]],
#        [[1, 1, "文字17"], [1, 1, "文字18"]]
# ]
# table_frame=[[[200, 98, 10, '基本信患']], [[239, 98, 2, '检验报告编号'], [240, 326, 2], [245, 593, 1], [249, 785, 5]], [[277, 99, 2, '号牌号码'], [277, 326, 2, '苏BG6IAC'], [282, 593, 1, '所有人'], [286, 785, 5]], [[330, 100, 2, '车辆类型'], [331, 327, 2, '轻型栏板货车'], [334, 593, 1, '品牌/型"'], [337, 785, 5]], [[366, 101, 2], [367, 327, 2], [371, 593, 1, '道路运龄证号', '品牌/型"'], [373, 785, 5]], [[405, 101, 2, '注册登记H期', '使用性质'], [407, 327, 2, '20I4-0-I0'], [410, 593, 1, 'HI厂H期', '道路运龄证号'], [411, 784, 2, '薛鹤'], [413, 1055, 3, '检验日期']], [[441, 100, 2, '藏解'], [443, 327, 2], [446, 593, 1, 'HI厂H期'], [447, 784, 2, '薛鹤'], [449, 1055, 3]], [[504, 99, 2, '检验类别'], [507, 327, 3], [509, 784, 2], [511, 1054, 3]], [[555, 98, 2], [558, 327, 8]], [[592, 98, 8], [598, 1131, 1], [598, 1257, 1]], [[630, 97, 2], [632, 326, 3], [634, 783, 2], [636, 1054, 1], [636, 1131, 1], [636, 1257, 1]], [[697, 98, 2], [699, 326, 3], [701, 783, 2], [702, 1054, 2], [702, 1361, 1]], [[734, 98, 10]], [[773, 98, 1, '序号'], [773, 238, 3], [775, 625, 1, '结果判定'], [775, 782, 3], [776, 1252, 1, '朱己宇']], [[810, 98, 1], [811, 238, 3], [812, 624, 1, '合格'], [812, 782, 3], [813, 1252, 1, '朱入宇', '朱己宇']], [[848, 98, 1], [849, 237, 3], [850, 624, 1, '合格', '合格'], [850, 781, 3], [851, 1252, 1, '朱飞宇', '朱入宇']], [[885, 98, 1], [885, 237, 3], [886, 623, 1, '合格'], [887, 781, 3], [888, 1252, 1, '朱飞宇', '朱飞宇']], [[923, 98, 1], [924, 236, 3], [925, 623, 1, '合格', '合格'], [925, 781, 3], [926, 1252, 1, '朱入宇', '朱飞宇']], [[960, 98, 1], [960, 236, 3], [961, 622, 1, '合格'], [962, 780, 3], [963, 1252, 1, '朱入宇']], [[998, 99, 1], [999, 235, 3], [1000, 622, 1, '合格'], [1000, 780, 3], [1002, 1252, 1]], [[1035, 98, 1], [1036, 235, 3], [1036, 622, 1, '合格'], [1037, 780, 3], [1039, 1252, 1]], [[1074, 98, 1], [1074, 235, 3], [1075, 622, 1], [1075, 780, 3], [1077, 1252, 1]], [[1110, 98, 1, '件号'], [1110, 234, 3], [1111, 621, 1, '检验结果'], [1112, 779, 2], [1113, 1052, 1, '合格/合格', '结果判定'], [1113, 1251, 1]], [[1148, 98, 1], [1149, 234, 3, '一轴空载削动率(8)/不平衡廖(%)'], [1150, 621, 1, '81.4/20.5'], [1151, 779, 2], [1152, 1052, 1, '合格/合格', '合格/合格'], [1153, 1251, 1]], [[1185, 98, 1], [1185, 234, 3, '二轴空载帕动率(M/不件衡率(%'], [1186, 621, 1, '68.6/8.0'], [1187, 779, 2], [1189, 1052, 1, '合格/合格', '合格/合格'], [1189, 1250, 1]], [[1187, 97, 1], [1185, 234, 3, '二轴空载帕动率(M/不件衡率(%'], [1186, 621, 1, '68.6/8.0'], [1187, 779, 2], [1189, 1052, 1, '合格/合格', '合格/合格'], [1189, 1250, 1]], [[1224, 97, 1], [1224, 234, 3, '整车制动率(%/驻车削动率()'], [1226, 621, 1, '76.I/26.5'], [1227, 779, 2], [1228, 1052, 1, '合格/合格'], [1229, 1250, 1]], [[1261, 97, 1], [1261, 234, 3, '住照灯左外灯远光发光强度(ed'], [1263, 621, 1, '15100', '76.I/26.5'], [1263, 779, 2], [1265, 1052, 1], [1266, 1250, 1]], [[1299, 96, 1], [1300, 234, 3, '前照灯右外灯远光发光强度(ab_'], [1301, 621, 1, 'I5100'], [1302, 779, 2], [1303, 1051, 1], [1304, 1250, 1]], [[1336, 96, 1], [1337, 234, 3], [1338, 621, 1], [1339, 779, 2], [1340, 1051, 1], [1341, 1249, 1]], [[1375, 96, 1], [1375, 234, 3], [1376, 621, 1], [1377, 778, 2], [1378, 1051, 1], [1380, 1249, 1]], [[1411, 96, 1], [1411, 234, 3], [1413, 621, 1], [1414, 778, 2], [1415, 1051, 1], [1416, 1249, 1]], [[1450, 96, 1], [1450, 234, 3], [1451, 621, 1], [1452, 778, 2], [1453, 1051, 1], [1455, 1249, 1]], [[1487, 96, 1], [1487, 234, 3], [1488, 621, 1], [1489, 778, 2], [1490, 1050, 1, 'I六、二维条码'], [1492, 1248, 1]], [[1525, 97, 1, 'I五、型议'], [1526, 234, 3], [1527, 621, 1], [1527, 778, 2], [1529, 1050, 1, 'I六、二维条码'], [1530, 1248, 1]], [[1562, 97, 7], [1566, 1050, 3]], [[1765, 98, 1], [1766, 233, 6], [1767, 1048, 3]], [[1881, 100, 1], [1882, 233, 9]]]

def generate_table_frame(frame, word_frame):
    lenlist = []
    for i in range(len(frame)):
        length = frame[i][-1][1] - frame[i][0][1]
        lenlist.append(length)
    kuanglen = max(lenlist)

    def sishewuru(x):
        x = int(x + 0.5)
        return x

    zonggeshu = 10
    for i in range(len(frame)):
        for j in range(len(frame[i]) - 1):
            frame[i][j].append(sishewuru(((frame[i][j + 1][1] - frame[i][j][1]) / kuanglen) * zonggeshu))
            # frame[i][j].append(frame[i][j + 1][1] - frame[i][j][1])

    for i in range(len(frame)):
        frame[i].pop()
    #
    thresholdy = 30
    thresholdx = 180
    table_frame = frame
    for i in range(len(frame)):
        for j in range(len(frame[i])):
            for k in range(len(word_frame)):
                if abs(frame[i][j][0] - word_frame[k][1]) <= thresholdy and abs(
                        frame[i][j][1] - word_frame[k][0]) <= thresholdx:
                    table_frame[i][j].append(word_frame[k][8])
    for i in range(len(table_frame)):
        for j in range(len(table_frame[i])):
            if len(table_frame[i][j])<4:
                table_frame[i][j].append('NONE')

    return table_frame
# print(table_frame)


def generate_table(table_frame, information):  # ['xxxx', 'xxxx', 'xxxx'] len()
    # 设置颜色
    style = xlwt.easyxf()
    # 设置边框
    borders = xlwt.Borders()  # Create Borders
    borders.left = 5
    borders.right = 5
    borders.top = 5
    borders.bottom = 5
    style.borders = borders
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('sheet 1')

    # 为指定单元格设置样式
    for i in range(len(table_frame)):  # 第i行
        k = 0
        for j in range(len(table_frame[i])-1):  #第j列
            sheet.write_merge(i, i, 0+k, 0+k+table_frame[i][j][2]-1, table_frame[i][j][3], style)
            k = k+table_frame[i][j][2]
            # sheet.write(i, j, table_frame[i][j][2])  # 第n行第i列写入内容
            print(i, j, table_frame[i][j][3])
        sheet.write_merge(i, i, 0 + k, 10, table_frame[i][len(table_frame[i])-1][3], style)
    for p in range(len(information)):
        sheet.write_merge(len(table_frame)+p, len(table_frame)+p, 0 , 10, information[p], style)
    wbk.save('./test/generate_table/机动车技术检验报告.xls')







