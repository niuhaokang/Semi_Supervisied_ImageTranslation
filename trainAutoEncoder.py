import torch
import os
from networks.lightweight_gan import Discriminator
from dataset import ImageDataSet

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = 'cuda:0'

    D = Discriminator(image_size=256).to(device)

    img = torch.rand((3, 3, 256, 256)).to(device)

    _,_,_ = D(img)
