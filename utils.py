import os
import cv2
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def EN(inputs):
    batch, c, h, w = inputs.shape
    entropies = torch.zeros(batch).to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))
    # transform value from 1 to 256
    inputs_int = (inputs * 255).int() + 1
    for i in range(batch):
        entropy = torch.zeros(c)
        for j in range(c):
            # calculate grey level and its pixel number
            value, counts = torch.unique(inputs_int[i, j, :, :], return_counts=True)
            p = counts / counts.sum()
            # only for pytorch 1.9.0 we can use torch.special
            # entropy[j] = torch.special.entr(p).sum()
            # for pytorch 1.7.0 use torch.log
            entropy[j] = (-p * torch.log(p)).sum()
        entropies[i] = torch.mean(entropy)

    return entropies


def std(inputs):
    # transform value from 1 to 256
    inputs = (inputs * 255).float() + 1
    std_batch = torch.std(inputs, dim=(2, 3))
    std_batch = torch.mean(std_batch, dim=-1)

    return std_batch


def spatial_freq(inputs):
    batch, c, h, w = inputs.shape
    inputs_int = (inputs * 255).int() + 1
    rf = torch.pow((inputs_int[:, :, 1:, :]-inputs_int[:, :, :h-1, :]), 2)
    rf = torch.sum(rf, dim=(2, 3))
    cf = torch.pow((inputs_int[:, :, :, 1:] - inputs_int[:, :, :, :w-1]), 2)
    cf = torch.sum(cf, dim=(2, 3))
    sf = torch.pow((rf+cf), 0.5)
    sf = torch.mean(sf, dim=-1)

    return sf


def features_grad(features):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    # Laplacian Kernel
    # kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    # Laplacian of Gaussian, LoG
    kernel = [[0, 1, 1, 2, 2, 2, 1, 1, 0], [1, 2, 4, 5, 5, 5, 4, 2, 1], [1, 4, 5, 3, 0, 3, 5, 4, 1],
              [2, 5, 3, -12, -24, -12, 3, 5, 2], [2, 5, 0, -24, -40, -24, 0, 5, 2], [2, 5, 3, -12, -24, -12, 3, 5, 2],
              [1, 4, 5, 3, 0, 3, 4, 4, 1], [1, 2, 4, 5, 5, 5, 4, 2, 1], [0, 1, 1, 2, 2, 2, 1, 1, 0]]

    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(device)
    features = features * 255
    _, c, _, _ = features.shape
    for i in range(int(c)):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


def features_grad_patch(features):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    # Laplacian Kernel
    # kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    # Laplacian of Gaussian, LoG
    kernel = [[0, 1, 1, 2, 2, 2, 1, 1, 0], [1, 2, 4, 5, 5, 5, 4, 2, 1], [1, 4, 5, 3, 0, 3, 5, 4, 1],
              [2, 5, 3, -12, -24, -12, 3, 5, 2], [2, 5, 0, -24, -40, -24, 0, 5, 2], [2, 5, 3, -12, -24, -12, 3, 5, 2],
              [1, 4, 5, 3, 0, 3, 4, 4, 1], [1, 2, 4, 5, 5, 5, 4, 2, 1], [0, 1, 1, 2, 2, 2, 1, 1, 0]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.to(device)
    features = features * 255
    batch, c, h, w = features.shape
    stride = 1
    size = kernel.shape[2]
    padding_h = int(((h - 1) * stride - h + size) / 2)
    padding_w = int(((w - 1) * stride - w + size) / 2)
    for i in range(int(c)):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=stride, padding=(padding_h, padding_w))
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


# input the root path of kaist dataset, return absolute path of images
def read_kaist(root_path):
    # root_path = .\kaist-cvpr15\images
    # extract every ir&vis directory root path
    path = []
    for root, dirs, files in os.walk(root_path, topdown=False):
        path.append(root)

    # extract every ir/vis directory path
    path_ir = list(filter(lambda path: path.find('lwir') >= 0, path))
    path_vis = list(filter(lambda path: path.find('visible') >= 0, path))
    # ir image path list
    ir_img = []
    for path in path_ir:
        ir_img.extend(glob.glob(os.path.join(path, '*.jpg')))
    # vis image path list
    vis_img = []
    for path in path_vis:
        vis_img.extend(glob.glob(os.path.join(path, '*.jpg')))

    # for kaist dataset, len of ir_img == vis_img, i.e. 95324
    # return absolute path of images
    return ir_img, vis_img


# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G