import os
import cv2
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from option import args
from utils import read_kaist


class TrainData(data.Dataset):
    '''
    only for gray image
    '''

    def __init__(self):
        super(TrainData, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.ir_path = args.ir_image_path
        self.vis_path = args.vis_image_path

        # Get the image name list
        self.ir_name = sorted(glob.glob(self.ir_path + '/*.png'))
        self.vis_name = sorted(glob.glob(self.vis_path + '/*.png'))

        self.patch = args.patch

    def __len__(self):
        assert len(self.ir_name) == len(self.vis_name)
        return len(self.ir_name)

    def __getitem__(self, idx):
        # Get image with gray scale
        ir_img = cv2.imread(self.ir_name[idx], cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(self.vis_name[idx], cv2.IMREAD_GRAYSCALE)
        ir_img = np.array(ir_img, dtype='float32') / 255.0
        vis_img = np.array(vis_img, dtype='float32') / 255.0

        ir_img = ir_img[:, :, np.newaxis]
        vis_img = vis_img[:, :, np.newaxis]

        # crop and augmentation
        ir_img, vis_img = self.crops(ir_img, vis_img)
        ir_img, vis_img = self.augmentation(ir_img, vis_img)

        # Permute the images to tensor format
        ir_img = np.transpose(ir_img, (2, 0, 1))
        vis_img = np.transpose(vis_img, (2, 0, 1))

        # Return image
        self.input_ir = ir_img.copy()
        self.input_vis = vis_img.copy()

        return self.input_ir, self.input_vis

    def crops(self, train_ir_img, train_vis_img):
        # Take random crops
        h, w, _ = train_ir_img.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_ir_img = train_ir_img[x: x + self.patch, y: y + self.patch]
        train_vis_img = train_vis_img[x: x + self.patch, y: y + self.patch]

        return train_ir_img, train_vis_img

    def augmentation(self, train_ir_img, train_vis_img):
        # Data augmentation
        if random.random() < 0.5:
            train_ir_img = np.flipud(train_ir_img)
            train_vis_img = np.flipud(train_vis_img)
        if random.random() < 0.5:
            train_ir_img = np.fliplr(train_ir_img)
            train_vis_img = np.fliplr(train_vis_img)
        rot_type = random.randint(1, 4)
        if random.random() < 0.5:
            train_ir_img = np.rot90(train_ir_img, rot_type)
            train_vis_img = np.rot90(train_vis_img, rot_type)

        return train_ir_img, train_vis_img


class TrainKaist(data.Dataset):
    '''
    Using Kaist dataset to train
    '''

    def __init__(self):
        super(TrainKaist, self).__init__()
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.ir_name, self.vis_name = read_kaist(args.kaist_path)
        self.patch = args.patch

    def __len__(self):
        assert len(self.ir_name) == len(self.vis_name)
        return len(self.ir_name)

    def __getitem__(self, idx):
        # Get image with gray scale
        ir_img = cv2.imread(self.ir_name[idx], cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(self.vis_name[idx], cv2.IMREAD_COLOR)

        # convert visible image from BGR to YCbCr
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCrCb)
        ir_img = np.array(ir_img, dtype='float32') / 255.0
        vis_img = np.array(vis_img, dtype='float32') / 255.0

        # extract Y channel of visible image
        ir_img = ir_img[:, :, np.newaxis]
        vis_img = vis_img[:, :, 0:1]

        # crop and augmentation
        ir_img, vis_img = self.crops(ir_img, vis_img)

        # Permute the images to tensor format
        ir_img = np.transpose(ir_img, (2, 0, 1))
        vis_img = np.transpose(vis_img, (2, 0, 1))

        # Return image
        self.input_ir = ir_img.copy()
        self.input_vis = vis_img.copy()

        return self.input_ir, self.input_vis

    def crops(self, train_ir_img, train_vis_img):
        # Take random crops
        h, w, _ = train_ir_img.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_ir_img = train_ir_img[x: x + self.patch, y: y + self.patch]
        train_vis_img = train_vis_img[x: x + self.patch, y: y + self.patch]

        return train_ir_img, train_vis_img


class TestData(data.Dataset):
    def __init__(self):
        super(TestData, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.ir_path = args.test_irimage_path
        self.vis_path = args.test_visimage_path

        # Get the image name list
        self.ir_name = sorted(glob.glob(self.ir_path + '/*.jpg'))
        self.vis_name = sorted(glob.glob(self.vis_path + '/*.jpg'))

    def __len__(self):
        assert len(self.ir_name) == len(self.vis_name)
        return len(self.ir_name)

    def __getitem__(self, idx):
        # Get image with gray scale
        ir_img = cv2.imread(self.ir_name[idx], cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(self.vis_name[idx], cv2.IMREAD_GRAYSCALE)
        ir_img = np.array(ir_img, dtype='float32') / 255.0
        vis_img = np.array(vis_img, dtype='float32') / 255.0

        ir_img = ir_img[:, :, np.newaxis]
        vis_img = vis_img[:, :, np.newaxis]

        # Permute the images to tensor format
        ir_img = np.transpose(ir_img, (2, 0, 1))
        vis_img = np.transpose(vis_img, (2, 0, 1))

        # Return image
        self.input_ir = ir_img.copy()
        self.input_vis = vis_img.copy()

        # Return image file name
        ir_file_name = self.ir_name[idx].split('.', 1)[0].rsplit("\\", 1)[1]
        vis_file_name = self.vis_name[idx].split('.', 1)[0].rsplit("\\", 1)[1]

        return self.input_ir, self.input_vis, ir_file_name, vis_file_name


class TestDataColor(data.Dataset):
    def __init__(self):
        super(TestDataColor, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.ir_path = args.test_irimage_path
        self.vis_path = args.test_visimage_path

        # Get the image name list
        self.ir_name = sorted(glob.glob(self.ir_path + '/*.png'))
        self.vis_name = sorted(glob.glob(self.vis_path + '/*.png'))

    def __len__(self):
        assert len(self.ir_name) == len(self.vis_name)
        return len(self.ir_name)

    def __getitem__(self, idx):
        # Get image with gray scale
        ir_img = cv2.imread(self.ir_name[idx], cv2.IMREAD_GRAYSCALE)
        vis_img = cv2.imread(self.vis_name[idx], cv2.IMREAD_COLOR)

        # convert visible image from BGR to YCbCr
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2YCrCb)
        ir_img = np.array(ir_img, dtype='float32') / 255.0
        vis_img = np.array(vis_img, dtype='float32') / 255.0

        # extract Y, Cb, Cr channel of visible image
        ir_img = ir_img[:, :, np.newaxis]
        vis_Y = vis_img[:, :, 0:1]
        vis_Cb = vis_img[:, :, 1]
        vis_Cr = vis_img[:, :, 2]

        # Permute the images to tensor format
        ir_img = np.transpose(ir_img, (2, 0, 1))
        vis_Y = np.transpose(vis_Y, (2, 0, 1))

        # Return image
        self.input_ir = ir_img.copy()
        self.input_vis = vis_Y.copy()

        # Return Cb and Cr channel
        self.vis_Cb = vis_Cb.copy()
        self.vis_Cr = vis_Cr.copy()

        # Return image file name
        ir_file_name = self.ir_name[idx].split('.', 1)[0].rsplit("\\", 1)[1]
        vis_file_name = self.vis_name[idx].split('.', 1)[0].rsplit("\\", 1)[1]

        return self.input_ir, self.input_vis, self.vis_Cb, self.vis_Cr, ir_file_name, vis_file_name
