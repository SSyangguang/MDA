import argparse

parser = argparse.ArgumentParser(description='Low light image fusion')
# Seed
parser.add_argument('--seed', type=int, default=2020, help='random seed')

# Data Acquisition
parser.add_argument('--ir_image_path', type=str, default='E:\project\yang\data/21_pairs_tno/ir',
                    help='infra image path of training')
parser.add_argument('--vis_image_path', type=str, default='E:\project\yang\data/21_pairs_tno/vis',
                    help='visible image path of training')
parser.add_argument('--test_irimage_path', type=str, default='E:\project\code\multitask2023\M3FD-simplified/ir/half-size',
                    help='infra image path of test')
parser.add_argument('--test_visimage_path', type=str, default='E:\project\code\multitask2023\M3FD-simplified/vis/half-size',
                    help='visible image path of test')
parser.add_argument('--kaist_path', type=str, default='/data/yg/data/kaist-cvpr15/images',
                    help='kaist dataset path')

# Training
parser.add_argument('--batch_size', type=int, default=24, help='batch size of fusion training')
parser.add_argument('--patch', type=int, default=192, help='patch size of fusion training')
# parser.add_argument('--loss_patch', type=int, default=16, help='patch size of loss calculation')
parser.add_argument('--epochs', type=int, default=2, help='epochs of fusion training')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate of fusion training')
parser.add_argument('--c_intensity', type=int, default=3000, help='calculate weight for intensity')
parser.add_argument('--c_detail', type=int, default=3000, help='calculate weight for detail')
parser.add_argument('--eta', type=float, default=0.5, help='eta in en and std normalization')

# Loss
parser.add_argument('--loss_alpha', type=int, default=0.2, help='alpha value for fusion model')
parser.add_argument('--loss_beta', type=float, default=0.167, help='beta value for fusion model')
parser.add_argument('--loss_gamma', type=int, default=20, help='gamma value for perceptual loss')
parser.add_argument('--loss_alpha2', type=float, default=1e-6, help='alpha value for perceptual loss')
parser.add_argument('--loss_style', type=float, default=1, help='value for style loss')
parser.add_argument('--loss_patch', type=float, default=2, help='value for patch loss')
parser.add_argument('--loss_patch_alpha', type=float, default=130, help='alpha value for patch loss')

# Log file
parser.add_argument('--log_dir', type=str, default='./fusion_train_log', help='fusion training log file path')
parser.add_argument('--model_path', type=str, default='./fusion_model/', help='fusion model path')
parser.add_argument('--model_path', type=str, default='/home/vipsl415/yg/code/inf_fusion2022/fusion_model/', help='fusion model path')
parser.add_argument('--model', type=str, default='fusion_model.pth', help='fusion model name')

parser.add_argument('--feature_num', type=int, default=32, help='number of features')

args = parser.parse_args(args=[])
