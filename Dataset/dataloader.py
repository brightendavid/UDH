"""
dataloader 读取原本数据集的src部分作为cover或者secrect
考虑到只需要将二值图像作为秘密信息加入,secrect可以考虑只有二值图，单通道
secrect图像使用生成的文本图，读取的数据集是用于cover图像
一般使用数据集DIV2K_valid_HR
使用?bit lsb 隐写作为传统嵌入方法
dataloade 基本没有问题
训练时，用于检测代码错误的数据集在20张以上，否则导致valdata数量为0，导致indexError.数组越界
"""

import os
import traceback

import cv2 as cv
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from LSB.lsb_data import gen_data

TEST = True


def gen_dataset(dir):
    paths = []
    for index, item in enumerate(os.listdir(dir)):
        img_path = os.path.join(dir, item)
        paths.append(img_path)
    return paths


def change_root_by_device(device=r"../data/DIV2K_train_HR"):
    """
    返回数据集路径
    """
    if device == 'sjw':
        # sjw 笔记本 移动硬盘路径
        data_root =r"../data/DIV2K_valid_HR"
    elif device == 'test':
        data_root = r"../data/DIV2K_valid_HR"
    else:
        data_root = r"../data/DIV2K_valid_HR"

    return data_root


class Mydataset:
    def __init__(self, transform=None, device="test", val_percent=0.1, train_val_test_mode='train'):
        self.root = change_root_by_device(device)
        self.image_list = gen_dataset(self.root)
        self.transform = transform
        self.train_val_test_mode = train_val_test_mode
        self.train_list, self.val_list = \
            train_test_split(self.image_list, test_size=val_percent,
                             train_size=1 - val_percent, random_state=1000)

    def __getitem__(self, index):
        if self.train_val_test_mode == "train":
            path = self.train_list[index]
        elif self.train_val_test_mode == "val":
            path = self.val_list[index]
        else:
            path = self.val_list[index]

        try:
            img = Image.open(path).convert('RGB')

            img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
            # cv.imshow("1",img)
            # cv.waitKey(0)
            img, secrect = add_tradition_watermark(img)

            if TEST:
                img = img[0:256, 0:256]
                secrect = secrect[0:256, 0:256]
            else:
                img = img[0:512, 0:512]
                secrect = secrect[0:512, 0:512]
            # img1 = Image.fromarray(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
            # secrect=Image.fromarray(cv.cvtColor(img1,cv.COLOR_RGB2GRAY))
            # todo 生成S
            # 可能有错，看pytorch版本,本机可以运行
            if self.transform:
                img_in = self.transform(img)
            else:
                img_in = transforms.Compose([
                    transforms.ToTensor()
                ])(img)
            if self.transform:
                img_out = self.transform(img)
            else:
                img_out = transforms.Compose([
                    transforms.ToTensor()
                ])(img)
            # cv.imshow("1",img)
            # cv.waitKey(0)
            secrect_in = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])(secrect)  # 张量化

            secrect_out = transforms.Compose([
                transforms.ToTensor()
            ])(secrect)  # 张量化

            return {'image': img_in, 'contain': img_out, 'secret_in': secrect_in, 'secret_out': secrect_out}
        except Exception as e:
            traceback.print_exc(e)

    def __len__(self):
        # 这里重要，不是原本的总list,而是分割后的list
        if self.train_val_test_mode == "train":
            return len(self.train_list)
        elif self.train_val_test_mode == "val":
            return len(self.val_list)


def add_tradition_watermark(src):
    """
    :param src:  原图
    :return:  加入4位LSB的含有水印的图像
    """
    src, secret = gen_data(src)
    return src, secret


def shou_tensor_img(tensor_img: torch.Tensor):
    """
    显示tensor
    """
    to_pil = torchvision.transforms.ToPILImage()
    img = tensor_img.cpu().clone()
    img = to_pil(img)
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    cv.imshow("img", img)
    cv.waitKey(0)


if __name__ == '__main__':
    testdataset = Mydataset(device="sjw")
    print(testdataset.__getitem__(0)["secret_in"])
    print(testdataset.__getitem__(0)["contain"].shape)
    shou_tensor_img(testdataset.__getitem__(0)["secret_in"])
    shou_tensor_img(testdataset.__getitem__(0)["secret_out"])
    shou_tensor_img(testdataset.__getitem__(0)["contain"])
    shou_tensor_img(testdataset.__getitem__(0)["image"])  # 有normalize，图像直接显示出现偏色是正常现象
    dataloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=1, num_workers=1)
    print(dataloader.__len__())
