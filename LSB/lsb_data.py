"""
宋佳维
2022/3/6
第四位LSB隐写

先弄成彩色的

lsb_data需要更新
"""
import cv2 as cv
import numpy as np

from LSB.Function import pic_reshape, w2PIL, gen_words

n = 3  # n表示第n位的LSB 隐写  256 22222222


class lsb:
    def __init__(self):
        pass

    def encode_pic_cai(self, pic, secret_pic):
        t = pic // pow(2, n)  # 去尾数
        t = t * pow(2, n)
        pic = t + pic % pow(2, n - 1)  # 还原，第n位置零
        secret_pic = secret_pic // 255
        pic = pic + secret_pic * pow(2, n - 1)
        return pic

    def decode_pic_cai(self, pic):
        t = pic // pow(2, n - 1)
        t = t % 2
        sum = t[:, :, 0] + t[:, :, 1] + t[:, :, 2]
        sum = np.where(sum < 1.5, 255, 0)
        cv.imwrite("1.png", sum)
        return sum


def lsb_function(src, src2):
    # 输出两张图像，返回一张桌面分辨率大小的图 src2为文字图像
    L = lsb()

    src = L.encode_pic_cai(src, src2)
    return src


def gen_data(src):
    word = gen_words()
    pic = w2PIL(word)
    src2 = pic_reshape(pic).astype('uint8')  # 此处数值类型必须要该，否则会造成int+float32造成失真
    src = lsb_function(src, src2)
    src2 = src2[:, :, 0]
    return src, src2


def pic2pic_shape(pic, src):
    # print(src)
    H, W = src.shape[0], src.shape[1]
    srcB = np.zeros((H, W, 3))
    row, col = H, W
    a, b, _ = pic.shape
    for i in range(row // a):
        for j in range(col // b):
            srcB[a * i:a * i + 21, b * j:b * j + b, :] = pic[:, :, :]
    return srcB


def add_lsb(data, secret):
    pic = w2PIL(secret)  # 文字转图像
    src2 = pic2pic_shape(pic, data).astype('uint8')  # 此处数值类型必须要该，否则会造成int+float32造成失真
    # print(src2.shape)
    src = lsb_function(data, src2)
    src2 = src2[:, :, 0]
    return src, src2


def out_secret(src):
    LL = lsb()
    secret = LL.decode_pic_cai(src)
    return secret


if __name__ == "__main__":
    src = cv.imread("../data/DIV2K_train_HR/0816.png")
    # src = cv.imread("./f1.png")
    src = cv.resize(src, (2048, 1024))
    t, t2 = gen_data(src)
    cv.imwrite("image.png", t)
    cv.imwrite("secret.png", t2)
    cv.imshow("imaeg", t)
    cv.imshow("secret", t2)
    out = out_secret(t)
    cv.imwrite("out_secret.png", out)
    cv.waitKey(2000)
