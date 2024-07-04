import base64
import os
import tkinter as tk
import tkinter.filedialog as tk_f  # 文件读取
import tkinter.messagebox  # 消息提醒
from tkinter import *

import xlwt
from PIL import Image, ImageTk
from tkinter import scrolledtext
from tkinter import ttk
import xlrd
from tkinter import filedialog
from ocr import *

ModelFlag = 0
imagePath = ''
photo = None
img = None

#选择文件
def openfile():
    sfname = filedialog.askopenfilename(title='选择Excel文件', filetypes=[('All Files', '*'), ('Excel', '*.xlsx') ])
    return sfname

"输入文件名，返回数据"
def readdata(sfname):
    # 读取表格数据
    book = xlrd.open_workbook(sfname)
    sheet1 = book.sheets()[0]
    nrows = sheet1.nrows
    print('表格总行数', nrows)
    ncols = sheet1.ncols
    print('表格总列数', ncols)

    values = []
    for i in range(nrows):
        row_values = sheet1.row_values(i)
        values.append(row_values)
    return values


def showdata(frame,data):
    # 定义树状图表格函数
    '''
    frame:容器
    data：数据，数据类型为列表

    '''

    nrows = len(data)
    ncols = len(data[0])
    columns = [""]
    for i in range(ncols):
        columns.append(str(i))
    heading = columns

    """
        定义Treeview
        self.Frame2为父容器
        columns为列名集合
        show="headings"表示显示表头
    """
    tree = ttk.Treeview(frame, columns=columns, show="headings")


    # 定义各列列宽及对齐方式
    for item in columns:
        tree.column(item, width=50, anchor="center")

    tree.heading(heading[0], text=heading[0])  #第一列的表头为空

    # 定义表头
    for i in range(1, len(columns)):
        tree.heading(heading[i], text=str(i))



    # 设置表格内容
    i = 0
    for v in data:
        v.insert(0, i + 1)    #第一列的显示内容(序号)
        tree.insert('', i, values=(v))
        i += 1

    # 放置控件，rel*表示使用相对定位，相对于父容器的定位
    # tree.place(relx=0, rely=0, relwidth=1, relheight=1)

    return tree


def createPage2():

    top = tk.Tk()
    top.title("基于计算机视觉的文档图片语义内容读取系统")
    top.geometry('900x700')  # 设置窗口大小
    top.resizable(0,0) #固定窗口大小

    selectedLabel = tk.Label(top,text='选择的文件', width=10, height=2)
    selectedLabel.pack()
    selectedLabel.place(x=30, y=26)

    path_var = tk.StringVar()
    path_text = Entry(top, bg='white', width=45, textvariable=path_var)
    path_text.pack()
    path_text.place(x=110, y=35)

    def selectImage():
        path_var.set('')
        file = tk_f.askopenfilename(parent=top, initialdir="D:/ClassProject",title='选择图片')
        global imagePath
        imagePath = file
        path_var.set(file)
        print(imagePath)
        showImage()

    selectedButton = tk.Button(top, text="选择图片", width='10', command=selectImage)
    selectedButton.pack()
    selectedButton.place(x=450, y=30)

    def choosenFunction():
        def choose1():
            global ModelFlag
            ModelFlag = 1
            print(ModelFlag)
        def choose2():
            global ModelFlag
            ModelFlag = 2
            print(ModelFlag)
        def choose3():
            global ModelFlag
            ModelFlag = 3
            print(ModelFlag)
        def choose4():
            global ModelFlag
            ModelFlag = 4
            print(ModelFlag)
        iv_default = IntVar()
        rb_default_Label = Label(top, text='检测方式')
        rb_default1 = Radiobutton(top, text='基于形态学操作法', value=1, variable=iv_default, relief=GROOVE, command=choose1)
        rb_default2 = Radiobutton(top, text='ctpn网络', value=2, variable=iv_default, relief=GROOVE, command=choose2)
        rb_default3 = Radiobutton(top, text='无表格识别', value=3, variable=iv_default, relief=GROOVE, command=choose3)
        rb_default4 = Radiobutton(top, text='带图文档识别', value=4, variable=iv_default, relief=GROOVE, command=choose4)
        iv_default.set(1)
        rb_default_Label.pack()
        rb_default1.pack()
        rb_default2.pack()
        rb_default3.pack()
        rb_default4.pack()
        rb_default_Label.place(x=35, y=95)
        rb_default1.place(x=130, y=95)
        rb_default2.place(x=300, y=95)
        rb_default3.place(x=470, y=95)
        rb_default4.place(x=640, y=95)
    choosenFunction()

    imageLabel = tk.Label(top, text='录入图片')
    imageLabel.pack()
    imageLabel.place(x=35, y=150)

    imageBoundary = tk.Label(top, text='', relief=GROOVE)
    imageBoundary.pack()
    imageBoundary.place(x=110, y=150, width=315, height=400)
    # root = tk.Tk()
    # root.title("打开文件")
    # root.geometry("600x400")
    # 打开文件并以树状表格形式显示
    def openshow():
        # global root
        filename = openfile()
        data = readdata(filename)
        tree = showdata(top, data)
        tree.place(relx=0.58, rely=0.2, relheight=0.59, relwidth=0.4)
    B1 = tk.Button(top, text="查看结果", command=openshow)
    B1.pack()
    B1.place(relx=0.5, rely=0.21)
    ############################
    showResultLabel = tk.Label(top, text='识别结果')
    showResultLabel.pack()
    showResultLabel.place(x=450, y=150)

    scr = tk.scrolledtext.ScrolledText(top)
    scr.pack()
    scr.place(x=520, y=150, width=315, height=400)

    def showImage():
        global photo
        global img
        img = Image.open(path_var.get())  # 打开图片
        img = img.resize((300, 400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(img)  # 用PIL模块的PhotoImage打开
        imglabel = Label(top, image=photo)
        imglabel.pack()
        imglabel.place(x=115, y=155)

    # 显示结果
    def getResult():
        scr.delete('1.0', 'end')
        scr.insert('end', '采用模型%d开始识别\n' % ModelFlag)
        print('采用模型%d开始识别' % ModelFlag)
        # 开始识别,并将识别的结果保存到列表中
        result = OCR(imagePath, ModelFlag)
        result.reverse()
        # 输出结果
        for text in result:
            scr.insert('end', text+'\n')

    # 初始化“开始识别按钮”，点击按钮触发函数getResult
    beginButton = tk.Button(top, text = "开始识别", command = getResult)
    beginButton.pack()
    beginButton.place(x=100, y=560)

    def clear():
        path_var.set('')
        scr.delete('1.0', 'end')
        imageBoundary = tk.Label(top, text='', relief=GROOVE)
        imageBoundary.pack()
        imageBoundary.place(x=110, y=150,width=315,height=295)

    clearButton = tk.Button(top, text="清 除", command=clear)
    clearButton.pack()
    clearButton.place(x=250,y=560)

    def end():
        sys.exit(0)
    quitButton = tk.Button(top, text=" 退 出 ", command=end)
    quitButton.pack()
    quitButton.place(x=380, y=560)

    mainloop()

createPage2()