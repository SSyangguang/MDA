import random
import torch

from option import args
from model import Train, Test, TestColor

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)


def train():
    train_ir_vis = Train()
    train_ir_vis.train()


def test():
    test_ir_vis = TestColor()
    test_ir_vis.test()


if __name__ == '__main__':
    test()
