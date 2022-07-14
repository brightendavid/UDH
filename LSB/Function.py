# -*- coding: utf-8 -*-
# 桌面分辨率大小 ： 1920*1080  改为2048 * 1024  是2的幂次，否则可能U-NET出问题
# 生成0和255的二值图像
import sys
sys.path.append('../')
import os
import string
import random
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2 as cv

H, W = 1024, 2048

TEST = True


def w2PIL(text):
    im = Image.new("RGB", (340, 21), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    if TEST:
        font = ImageFont.truetype(os.path.join("fonts", "simsun.ttc"), 18)  # windows
    else:
        # linux下
        font = ImageFont.truetype(
            r"/home/liu/anaconda3/envs/deeplearn/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/cmss10.ttf",
            18)

    dr.text((0, 0), text, font=font, fill="#000000")

    # im.show()
    # im.save("t.png")
    im = np.array(im)
    im = np.where(im > 200, 0, 255)
    cv.imwrite("Word2Pic/t.png", im)
    return im


def pic_reshape(src):
    """
    把文字图片src 反复多次，指导其为桌面分辨率大小
    :param src:   文字图片
    :return:   文字图片  分辨率变大
    """
    srcB = np.zeros((H, W, 3))
    row, col = H, W
    a, b, _ = src.shape
    for i in range(row // a):
        for j in range(col // b):
            srcB[a * i:a * i + 21, b * j:b * j + b, :] = src[:, :, :]
    return srcB


def gen_words():
    """
    随机生成
    """
    name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(6))  # 6位字母数字作为用户名
    time = ''.join(random.choice(string.digits) for _ in range(12))  # 12位数字作为时间

    ip = ''.join(random.choice(string.digits) for _ in range(3)) + '.'
    ip = ip + ''.join(random.choice(string.digits) for _ in range(3)) + '.'
    ip = ip + ''.join(random.choice(string.digits) for _ in range(3)) + '.'
    ip = ip + ''.join(random.choice(string.digits) for _ in range(3))  # 4段的3位数字作为Ip地址
    word = name + '-' + ip + '-' + time

    return word


if __name__ == '__main__':
    # text = u"brighten-10.230.14.215-202202221536"
    # pic = w2PIL(text)
    # print(pic.shape)
    # pic = pic_reshape(pic)
    print(gen_words())
    word = gen_words()
    pic = w2PIL(word)
    print(pic.shape)
    pic = pic_reshape(pic)
    cv.imwrite("test.png", pic)
    # cv.imshow("1", pic)
    # cv.waitKey(0)
