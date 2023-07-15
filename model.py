import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from option import args
from dataloader import TrainData, TestData, TrainKaist, TestDataColor
from net import FusionNet, vgg16
from loss import ssim, ms_ssim, SSIM, MS_SSIM, CharbonnierLoss, Perloss
from utils import EN, std, features_grad, features_grad_patch, gram

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)


class Train(object):
    def __init__(self):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.num_epochs = args.epochs
        self.batch = args.batch_size
        self.lr = args.lr
        # load data and transform image to tensor and normalize
        self.train_set = TrainKaist()
        self.train_loader = data.DataLoader(self.train_set, batch_size=self.batch,
                                            shuffle=True, num_workers=8, pin_memory=True)

        self.fusion_model = FusionNet().to(self.device)
        # load vgg model
        self.vgg = vgg16().to(self.device)
        self.vgg.load_state_dict(torch.load('vgg16.pth'))
        self.optimizer = Adam(self.fusion_model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.mse = nn.MSELoss(reduction='mean').to(self.device)
        self.kldiv = nn.KLDivLoss(reduction='batchmean', log_target=True).to(self.device)

        self.ssim = SSIM(channel=1)
        self.ms_ssim = MS_SSIM(data_range=255, size_average=True, channel=1)
        self.c_loss = CharbonnierLoss(eps=1e-3)
        self.per_loss = Perloss(gamma=args.loss_gamma)
        self.loss = []

    def train(self):
        writer = SummaryWriter(log_dir=args.log_dir, filename_suffix='train_loss')
        net = self.fusion_model

        # build folder
        if not os.path.exists(args.model_path):
            os.mkdir(args.model_path)
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        # load pre-trained fusion model
        if os.path.exists(args.model_path + args.model):
            print('Loading pre-trained model')
            state = torch.load(args.model_path + args.model)
            net.load_state_dict(state['model'])

        for epoch in range(self.num_epochs):
            loss_total_epoch = []

            for batch, (ir, vis) in enumerate(self.train_loader):
                img1 = ir.to(self.device)
                img2 = vis.to(self.device)

                # fuse image
                fusion, ir_fea, vis_fea, fusion_fea = net(img1, img2)
                fusion = fusion.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    # extract feature through vgg-16
                    feature_1 = torch.cat((img1, img1, img1), dim=1)
                    feature_1 = self.vgg(feature_1)
                    feature_2 = torch.cat((img2, img2, img2), dim=1)
                    feature_2 = self.vgg(feature_2)

                    for i in range(0, len(feature_1)-3):

                        # calculate intensity weight, only 1 to 2 pooling layers
                        if i >= 0:
                            en_1 = EN(feature_1[i])
                            en_2 = EN(feature_2[i])
                            std_1 = std(feature_1[i])
                            std_2 = std(feature_2[i])
                            en_std_1 = en_1 + args.loss_beta * std_1
                            en_std_2 = en_2 + args.loss_beta * std_2

                            if i == 0:
                                inten_1 = torch.unsqueeze(en_std_1, dim=-1)
                                inten_2 = torch.unsqueeze(en_std_2, dim=-1)
                            else:
                                inten_1 = torch.cat((inten_1, torch.unsqueeze(en_std_1, dim=-1)), dim=-1)
                                inten_2 = torch.cat((inten_2, torch.unsqueeze(en_std_2, dim=-1)), dim=-1)

                        # calculate detail weight, only 1 to 2 pooling layers
                        if i >= 0:
                            grad_1 = torch.mean(features_grad(feature_1[i]).pow(2), dim=[1, 2, 3])
                            grad_2 = torch.mean(features_grad(feature_2[i]).pow(2), dim=[1, 2, 3])

                            if i == 0:
                                detail_1 = torch.unsqueeze(grad_1, dim=-1)
                                detail_2 = torch.unsqueeze(grad_2, dim=-1)
                            else:
                                detail_1 = torch.cat((detail_1, torch.unsqueeze(grad_1, dim=-1)), dim=-1)
                                detail_2 = torch.cat((detail_2, torch.unsqueeze(grad_2, dim=-1)), dim=-1)

                    inten_weight_1 = torch.mean(inten_1, dim=-1) / args.c_intensity
                    inten_weight_2 = torch.mean(inten_2, dim=-1) / args.c_intensity
                    inten_weight_list = torch.cat((inten_weight_1.unsqueeze(-1), inten_weight_2.unsqueeze(-1)), -1)
                    inten_weight_list = F.softmax(inten_weight_list, dim=-1).to(self.device)

                    detail_weight_1 = torch.mean(detail_1, dim=-1) / args.c_detail
                    detail_weight_2 = torch.mean(detail_2, dim=-1) / args.c_detail
                    detail_weight_list = torch.cat((detail_weight_1.unsqueeze(-1), detail_weight_2.unsqueeze(-1)), -1)
                    detail_weight_list = F.softmax(detail_weight_list, dim=-1).to(self.device)


                self.optimizer.zero_grad()

                # calculate sim loss
                ssim_img1 = (1 - self.ms_ssim(fusion, img1))
                ssim_img2 = (1 - self.ms_ssim(fusion, img2))
                loss_sim = detail_weight_list[:, 0] * ssim_img1 + detail_weight_list[:, 1] * ssim_img2
                loss_sim = torch.mean(loss_sim)

                # calculate mse loss
                loss_mse = inten_weight_list[:, 0] * self.mse(fusion, img1) + inten_weight_list[:, 1] * self.mse(fusion,
                                                                                                                 img2)
                loss_mse = torch.mean(loss_mse)
                print('loss ssim: ', loss_sim)
                print('loss mse: ', loss_mse)

                # calculate perceptual loss
                loss_per = self.per_loss(fusion_fea, ir_fea, vis_fea)
                loss_per = torch.mean(loss_per)
                print('per loss: ', loss_per)

                # calculate style loss
                style_vgg = torch.cat((img1, img1, img1), dim=1)
                style_features = self.vgg(style_vgg)
                style_gram = [gram(fmap) for fmap in style_features]
                fusion_vgg = torch.cat((fusion, fusion, fusion), dim=1)
                fusion_features = self.vgg(fusion_vgg)
                fusion_gram = [gram(fmap) for fmap in fusion_features]
                loss_style = 0.0
                for j in range(2):
                    loss_style += self.kldiv(fusion_gram[j].softmax(dim=-1).log(), style_gram[j].softmax(dim=-1).log())
                print('style loss: ', loss_style)

                # calculate patch loss
                size = 15
                stride = 1
                b_patch, c_patch, h_patch, w_patch = fusion.shape
                padding_h = int(((h_patch - 1) * stride - h_patch + size) / 2)
                padding_w = int(((w_patch - 1) * stride - w_patch + size) / 2)
                C1 = 6.25
                window = torch.ones((1, 1, size, size)) / (size * size)
                window = window.to(self.device)

                # calculate the intensity sum of the window
                mean_ir = F.conv2d(img1, window, stride=stride, padding=(padding_h, padding_w))
                mean_vis = F.conv2d(img2, window, stride=stride, padding=(padding_h, padding_w))
                mean_fusion = F.conv2d(fusion, window, stride=stride, padding=(padding_h, padding_w))

                mean_ir_2 = F.conv2d(torch.pow(img1, 2), window, stride=stride, padding=(padding_h, padding_w))
                mean_vis_2 = F.conv2d(torch.pow(img2, 2), window, stride=stride, padding=(padding_h, padding_w))
                mean_fusion_2 = F.conv2d(torch.pow(fusion, 2), window, stride=stride, padding=(padding_h, padding_w))

                # calculate the standard deviation of the window
                var_ir = mean_ir_2 - torch.pow(mean_ir, 2)
                var_vis = mean_vis_2 - torch.pow(mean_vis, 2)
                var_fusion = mean_fusion_2 - torch.pow(mean_fusion, 2)

                mean_ir_f = F.conv2d(img1 * fusion, window, stride=stride, padding=(padding_h, padding_w))
                mean_vis_f = F.conv2d(img2 * fusion, window, stride=stride, padding=(padding_h, padding_w))

                sigma_ir_f = mean_ir_f - mean_ir * mean_fusion
                sigma_vis_f = mean_vis_f - mean_vis * mean_fusion

                C1 = torch.ones(sigma_ir_f.shape) * C1
                C1 = C1.to(self.device)

                # calculate ssim
                ssim_ir_f = (2 * sigma_ir_f + C1) / (var_ir + var_fusion + C1)
                ssim_vis_f = (2 * sigma_vis_f + C1) / (var_vis + var_fusion + C1)

                # calculate the ssim weight
                grad_ir = F.conv2d(torch.pow(features_grad_patch(img1), 2), window, stride=stride, padding=(padding_h, padding_w))
                grad_vis = F.conv2d(torch.pow(features_grad_patch(img2), 2), window, stride=stride, padding=(padding_h, padding_w))
                grad_ir = F.conv2d(torch.pow(grad_ir, 1 / 2), window, stride=stride, padding=(padding_h, padding_w)) / args.c_detail
                grad_vis = F.conv2d(torch.pow(grad_vis, 1 / 2), window, stride=stride, padding=(padding_h, padding_w)) / args.c_detail
                patch_grad_patch = torch.cat((grad_ir, grad_vis), 1)
                patch_grad_patch = F.softmax(patch_grad_patch, dim=1).to(self.device)

                # calculate the ssim loss
                ssim_patch = patch_grad_patch[:, 0:1, :, :] * (1 - ssim_ir_f) + patch_grad_patch[:, 1:2, :, :] * (1 - ssim_vis_f)
                ssim_patch = torch.mean(ssim_patch)

                # calculate intensity weight
                int_ir = (mean_ir + args.loss_beta * var_ir) / args.c_intensity
                int_vis = (mean_vis + args.loss_beta * var_vis) / args.c_intensity
                patch_int_patch = torch.cat((int_ir, int_vis), 1)
                patch_int_patch = F.softmax(patch_int_patch, dim=1).to(self.device)

                # calculate the mse loss
                loss_mse_patch = nn.MSELoss(reduction='none').to(self.device)
                mse_ir = F.conv2d(loss_mse_patch(fusion, img1), window, stride=stride, padding=(padding_h, padding_w))
                mse_vis = F.conv2d(loss_mse_patch(fusion, img2), window, stride=stride, padding=(padding_h, padding_w))

                # calculate the weighted mse loss
                mse_patch = patch_int_patch[:, 0:1, :, :] * mse_ir + patch_int_patch[:, 1:2, :, :] * mse_vis
                mse_patch = torch.mean(mse_patch)

                loss_patch = ssim_patch + args.loss_alpha * mse_patch
                print('loss_patch:', loss_patch)

                # calculate total loss
                loss_total = loss_sim + args.loss_alpha * loss_mse + args.loss_alpha2 * loss_per + args.loss_style * loss_style + args.loss_patch * loss_patch
                loss_total_epoch.append(loss_total.item())

                loss_total.backward()
                self.optimizer.step()

            self.loss.append(np.mean(loss_total_epoch))
            print('epoch: %s, loss: %s' % (epoch, np.mean(loss_total_epoch)))

            state = {
                'model': self.fusion_model.state_dict(),
                'train_loss': self.loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            torch.save(state, args.model_path + args.model)
            if epoch % 1 == 0:
                torch.save(state, args.model_path + str(epoch) + '.pth')

            writer.add_scalar('loss', np.mean(loss_total_epoch), epoch)

        fig_sim, axe_sim = plt.subplots()
        axe_sim.plot(self.loss)
        fig_sim.savefig('train_loss_curve.png')

        print('Training finished')


class Test(object):
    def __init__(self, batch_size=1):
        super(Test, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs
        self.feature_num = args.feature_num

        # load data and transform image to tensor and normalize
        self.test_set = TestData()
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0, pin_memory=True)

        self.nb_filter = [2*self.feature_num, 3*self.feature_num, 4*self.feature_num, 5*self.feature_num]

        self.fusion_model = FusionNet().to(self.device)
        self.fusion_state = torch.load(args.model_path + args.model, map_location='cuda:0')
        self.fusion_model.load_state_dict(self.fusion_state['model'])

    def test(self):
        fusion_model = self.fusion_model
        fusion_model.eval()

        for batch, (ir_img, vis_img, ir_name, vis_name) in enumerate(self.test_loader):
            ir_img = ir_img.to(self.device)
            vis_img = vis_img.to(self.device)

            outputs, _, _, _ = fusion_model(ir_img, vis_img)

            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            outputs_scale = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            outputs_scale = (outputs_scale * 255).astype(np.int)
            cv2.imwrite('./fusion_result/gray/%s.jpg' % ir_name[0], outputs_scale)


class TestColor(object):
    def __init__(self, batch_size=1):
        super(TestColor, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.epochs
        self.feature_num = args.feature_num

        # load data and transform image to tensor and normalize
        self.test_set = TestDataColor()
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0, pin_memory=True)

        self.nb_filter = [2*self.feature_num, 3*self.feature_num, 4*self.feature_num, 5*self.feature_num]

        self.fusion_model = FusionNet().to(self.device)
        self.fusion_state = torch.load(args.model_path + args.model, map_location='cuda:0')
        self.fusion_model.load_state_dict(self.fusion_state['model'])

    def test(self):
        fusion_model = self.fusion_model
        fusion_model.eval()

        for batch, (ir_img, vis_img, vis_Cb, vis_Cr, ir_name, vis_name) in enumerate(self.test_loader):
            ir_img = ir_img.to(self.device)
            vis_img = vis_img.to(self.device)

            outputs, _, _, _ = fusion_model(ir_img, vis_img)

            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            # save gray fusion image
            outputs_scale = (outputs - outputs.min()) / (outputs.max() - outputs.min())
            outputs_scale = (outputs_scale * 255).astype(np.int)

            # save color image
            color = np.stack((outputs_scale, vis_Cb.squeeze() * 255, vis_Cr.squeeze() * 255), axis=2)
            color = color.astype('uint8')
            color = cv2.cvtColor(color, cv2.COLOR_YCrCb2BGR)
            cv2.imwrite('E:\project\code\multitask2023/M3FD-simplified/fusion/%s.jpg' % ir_name[0], color)
