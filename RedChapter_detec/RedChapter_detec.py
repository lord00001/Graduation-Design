
import cv2
import numpy as np
import os

def RedChapter_detec(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([100, 30, 100])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask=mask)
    r, g, b = cv2.split(res)
    r_num = 0
    for i in b:
        for j in i:
            if j > 170:
                r_num += 1
    # cv2.namedWindow('res', 50)
    # cv2.resizeWindow('res', 300, 400)
    # cv2.namedWindow('img', 50)
    # cv2.resizeWindow('img', 300, 400)
    # cv2.namedWindow('mask', 50)
    # cv2.resizeWindow('mask', 300, 400)
    # cv2.imshow('img', img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("res", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if(r_num>30):
        result = "该图片有红章"
        cv2.imwrite('./test/RedChapter_handwriting_output/RedChapter.jpg', res)
        return result
    else:
        result = "该图片没有红章"
        return result
