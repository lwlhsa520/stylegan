import glob
import os
import re
import time
from pathlib import Path

import PIL
from tqdm import tqdm

import legacy
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch.nn.functional as F

import dnnlib

# BOX_COLOR = (255, 0, 0)  # Red
# TEXT_COLOR = (255, 255, 255)  # White
from torch_utils.aug_util import xywh2x0y0x1y1
from torch_utils.common import VGGLoss, Perceptual_loss134
from torch_utils.custom_ops import bbox_mask
from training.SimNet import SimGenerator
from training.dataset import ImageFolderDataset


def main():
    batch_size = 1
    seed = 20
    out = "./samples/"
    device = torch.device('cuda')
    device2 = torch.device('cpu')
    # _label = generate_labels(batch_gpu, seed=10)
    # my_label = torch.as_tensor(np.loadtxt("./data/data_0_1.txt"), device=device)
    # if len(my_label)<128:
    #     my_label = torch.cat([my_label, torch.zeros([128-len(my_label), 5], device=device)], 0)
    # print(my_label)
    test_set = ImageFolderDataset(path='../data/dataset4', use_labels=False, bbox_dim=256, max_size=None, xflip=False,
                                  aug=False)
    num = len(test_set)
    dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True, num_workers=3,
                                             prefetch_factor=2)

    C, H, W = test_set[0][0].shape

    network_pkl = './results/noDmask/00000-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl'
    # network_pkl = '../res/stylegan_ori1/00000-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl'
    # network_pkl = '../res/stylegan/00005-dataset-auto1-noaug/network-snapshot-000800.pkl'
    # network_pkl = '../res/stylegan/stylegan_init/00100-dataset-auto1-batch8/network-snapshot-000600.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if not os.path.exists(os.path.join(out, 'dataset4_crops')):
        os.makedirs(os.path.join(out, 'dataset4_crops'))
    for index, data in enumerate(tqdm(dataloader)):
        images, masks, bboxs = data

        bboxs = xywh2x0y0x1y1(bboxs)

        bboxs[:,:,1:] = (bboxs[:, :, 1:] * 256)
        bboxs = bboxs.to(torch.uint8).cpu().numpy()

        for j, box in enumerate(bboxs[0]):
            if box[0] != 0:
                img = torch.zeros_like(images[0, 0])
                img[box[1]:box[3], box[2]:box[4]] = images[0, 0, box[1]:box[3], box[2]:box[4]]
                PIL.Image.fromarray(img.cpu().numpy()).convert('L').save(
                    os.path.join(out, f'dataset4_crops/{(index * 1000 + j):07d}.png'))

if __name__ == "__main__":
    main()