import os

import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms.functional import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def affine_rotating(img_torch,ang=360):
    """
    ang:角度为-180-180之间
    """
    # 旋转
    angle =(ang * math.pi / 180)
    theta = torch.tensor([
        [math.cos(angle), math.sin(-angle), 0, 0],
        [math.sin(angle), math.cos(angle), 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid).cuda()
    new_img_torch = output[0].cuda()
    return new_img_torch


def affine_big22(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.5, 0, 0, 0],
        [0, 0.5, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch

def affine_big23(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.5, 0, 0, 0],
        [0, 0.3, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_big50(img_torch):
    # 放大 5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed
    theta = torch.tensor([
        [0.2, 0, 0, 0],
        [0, 0.1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_small22(img_torch):
    # 缩小  5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed   ,尺度可以变化
    theta = torch.tensor([
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch

def affine_small23(img_torch):
    # 缩小  5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed   ,尺度可以变化
    theta = torch.tensor([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def affine_small32(img_torch):
    # 缩小  5维  3*4   4维3*3    3维 2*3  1 means this aix is not changed   ,尺度可以变化
    theta = torch.tensor([
        [3, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size()).cuda()
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    return new_img_torch


def change(img):
    # 改变对比度，亮度，色度,只能对image使用，返回Image
    hue_factor = 0.2
    bright_factor = 0.9
    con_factor = 0.7
    img = adjust_brightness(img, bright_factor)
    img = adjust_contrast(img, con_factor)
    img = adjust_hue(img, hue_factor)
    return img


def get_rnd_brightness(rnd_bri, rnd_hue, batch_size):
    # 使用正态分布，模拟曝光和对比度的变化
    rnd_hueness = torch.distributions.uniform.Uniform(-rnd_hue, rnd_hue).sample([batch_size, 3, 1, 1])
    rnd_brightness = torch.distributions.uniform.Uniform(-rnd_bri, rnd_bri).sample([batch_size, 1, 1, 1])
    return rnd_hueness + rnd_brightness



if __name__ == '__main__':
    img_path = r'../data/DIV2K_train_HR/0801.png'
    img=Image.open(img_path)
    # plt.imshow(img)
    # plt.show()

    img_torch = transforms.ToTensor()(img)
    print(img_torch.shape)
    img_torch = img_torch.unsqueeze(0)
    # img_torch = torch.randn(1, 1, 192, 160)
    # t = get_rnd_brightness(0.4, 0.3, img_torch.shape[0])
    # print(t.shape)
    # print("t", t)  # t为-1-+1的浮点数，直接对tensor操作
    # print(img_torch.shape)
    # img = img_torch + t
    # img = affine_rotating(img_torch)
    img = F.interpolate(img_torch, scale_factor=(0.4, 0.2), mode='bilinear')
    print(img.shape)
    # # img = change(Image.open(img_path))
    # img = affine_big(img)
    # print(img.shape)
    img = img.squeeze(0)
    plt.imshow(img.numpy().transpose(1, 2, 0))
    # # plt.imshow(img)
    plt.show()
