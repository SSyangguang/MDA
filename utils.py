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
    # kernel = [[0, 1, 1, 1 / 2, 1 / 2, 1 / 2, 1, 1, 0], [1, 1 / 2, 1 / 4, 1 / 5, 1 / 5, 1 / 5, 1 / 4, 1 / 2, 1], [1, 1 / 4, 1 / 5, 1 / 3, 0, 1 / 3, 1 / 5, 1 / 4, 1],
    #           [1 / 2, 1 / 5, 1 / 3, -1 / 12, -1 / 24, -1 / 12, 1 / 3, 1 / 5, 1 / 2], [1 / 2, 1 / 5, 0, -1 / 24, -1 / 40, -1 / 24, 0, 1 / 5, 1 / 2], [1 / 2, 1 / 5, 1 / 3, -1 / 12, -1 / 24, -1 / 12, 1 / 3, 1 / 5, 1 / 2],
    #           [1, 1 / 4, 1 / 5, 1 / 3, 0, 1 / 3, 1 / 4, 1 / 4, 1], [1, 1 / 2, 1 / 4, 1 / 5, 1 / 5, 1 / 5, 1 / 4, 1 / 2, 1], [0, 1, 1, 1 / 2, 1 / 2, 1 / 2, 1, 1, 0]]
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
    # kernel = [[0, 1, 1, 1 / 2, 1 / 2, 1 / 2, 1, 1, 0], [1, 1 / 2, 1 / 4, 1 / 5, 1 / 5, 1 / 5, 1 / 4, 1 / 2, 1],
    #           [1, 1 / 4, 1 / 5, 1 / 3, 0, 1 / 3, 1 / 5, 1 / 4, 1],
    #           [1 / 2, 1 / 5, 1 / 3, -1 / 12, -1 / 24, -1 / 12, 1 / 3, 1 / 5, 1 / 2],
    #           [1 / 2, 1 / 5, 0, -1 / 24, -1 / 40, -1 / 24, 0, 1 / 5, 1 / 2],
    #           [1 / 2, 1 / 5, 1 / 3, -1 / 12, -1 / 24, -1 / 12, 1 / 3, 1 / 5, 1 / 2],
    #           [1, 1 / 4, 1 / 5, 1 / 3, 0, 1 / 3, 1 / 4, 1 / 4, 1],
    #           [1, 1 / 2, 1 / 4, 1 / 5, 1 / 5, 1 / 5, 1 / 4, 1 / 2, 1], [0, 1, 1, 1 / 2, 1 / 2, 1 / 2, 1, 1, 0]]
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


class SplitConcatImg():
    '''
    split an image to patches and concatenate patches to an image
    '''
    def __init__(self, img, save_path, patch=128):
        self.img = img
        self.path = save_path
        self.patch = patch
        self.h, self.w = img.shape
        self.h_retain = self.h % self.patch
        self.w_retain = self.w % self.patch
        self.h_num = self.h // self.patch if self.h_retain == 0 else self.h // self.patch + 1
        self.w_num = self.w // self.patch if self.w_retain == 0 else self.w // self.patch + 1

    def split_image(self):
        # save patch without last row and column
        for i in range(self.h_num - 1):
            for j in range(self.w_num - 1):
                patch_img = self.img[i * self.patch: (i + 1) * self.patch, j * self.patch: (j + 1) * self.patch]
                cv2.imwrite(os.path.join(self.path, 'patch_row_%i_col_%i.jpg' % (i, j)), patch_img)

        # save last column patch without last row
        for i in range(self.h_num - 1):
            patch_img = self.img[i * self.patch: (i + 1) * self.patch, (self.w_num - 1) * self.patch:]
            cv2.imwrite(os.path.join(self.path, 'patch_row_%i_col_%i.jpg' % (i, self.w_num - 1)), patch_img)

        # save last row patch
        for i in range(self.w_num):
            patch_img = self.img[(self.h_num - 1) * self.patch:, i * self.patch: (i + 1) * self.patch]
            cv2.imwrite(os.path.join(self.path, 'patch_row_%i_col_%i.jpg' % (self.h_num - 1, i)), patch_img)

    def concat_image(self):
        # compose patch from row to column
        output = np.zeros((self.h, self.w), dtype=int)
        # compose patch without last row and column
        for i in range(self.h_num - 1):
            for j in range(self.w_num - 1):
                patch_img = cv2.imread(os.path.join(self.path, 'patch_row_%i_col_%i.jpg' % (i, j)), cv2.IMREAD_GRAYSCALE)
                output[i * self.patch: (i + 1) * self.patch, j * self.patch: (j + 1) * self.patch] = patch_img

        # compose last column without last row
        for i in range(self.h_num - 1):
            patch_img = cv2.imread(os.path.join(self.path, 'patch_row_%i_col_%i.jpg' % (i, self.w_num - 1)), cv2.IMREAD_GRAYSCALE)
            output[i * self.patch: (i + 1) * self.patch, (self.w_num - 1) * self.patch:] = patch_img

        # compose last row patch
        for i in range(self.w_num):
            patch_img = cv2.imread(os.path.join(self.path, 'patch_row_%i_col_%i.jpg' % (self.h_num - 1, i)), cv2.IMREAD_GRAYSCALE)
            output[(self.h_num - 1) * self.patch:, i * self.patch: (i + 1) * self.patch] = patch_img

        cv2.imwrite(os.path.join(self.path, 'output.jpg'), output)


# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G