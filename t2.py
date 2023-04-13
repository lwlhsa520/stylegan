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
from t3 import COLOR_MAP
from torch_utils.aug_util import xywh2x0y0x1y1
from torch_utils.common import VGGLoss, Perceptual_loss134
from torch_utils.custom_ops import bbox_mask
from training.SimNet import SimGenerator
from training.dataset import ImageFolderDataset


def main():
    batch_size = 1
    seed = 20
    out = "./samples/sample_T_dmask2"
    device = torch.device('cuda')
    device2 = torch.device('cpu')
    # _label = generate_labels(batch_gpu, seed=10)
    # my_label = torch.as_tensor(np.loadtxt("./data/data_0_1.txt"), device=device)
    # if len(my_label)<128:
    #     my_label = torch.cat([my_label, torch.zeros([128-len(my_label), 5], device=device)], 0)
    # print(my_label)
    test_set = ImageFolderDataset(path='../data/dataset', use_labels=False, bbox_dim=256, max_size=None, xflip=False,
                                  aug=False)
    num = len(test_set)
    dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=3,
                                             prefetch_factor=2)

    C, H, W = test_set[0][0].shape

    network_pkl = './results/test4/00000-stylegan2-dataset4-gpus1-batch16/best_model.pkl'
    # network_pkl = '../res/stylegan_ori1/00000-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl'
    # network_pkl = '../res/stylegan/00005-dataset-auto1-noaug/network-snapshot-000800.pkl'
    # network_pkl = '../res/stylegan/stylegan_init/00100-dataset-auto1-batch8/network-snapshot-000600.pkl'
    with dnnlib.util.open_url(network_pkl) as f:

        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    if not os.path.exists(out):
        os.mkdir(out)
    if not os.path.exists(os.path.join(out, 'images')):
        os.mkdir(os.path.join(out, 'images'))
    if not os.path.exists(os.path.join(out, 'masks')):
        os.mkdir(os.path.join(out, 'masks'))
    if not os.path.exists(os.path.join(out, 'crops')):
        os.mkdir(os.path.join(out, 'crops'))
    for i in range(1):
        for index, data in enumerate(tqdm(dataloader)):
            real_images, real_masks, real_bbox = data
            bboxs = real_bbox.to(device)
            z = torch.from_numpy(np.random.RandomState(i*num+index).randn(batch_size, G.z_dim)).to(device)
            gen_images, gen_masks, _, gen_mid_masks = G(z, bboxs)
            gen_images = (gen_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            gen_mid_masks = (F.interpolate(gen_mid_masks, (256, 256), mode='nearest').sum(1, keepdim=True) * 255).clamp(0,255).to(
                torch.uint8).cpu().numpy()
            gen_masks = (gen_masks.sign() * 255).clamp(0, 255).to(torch.uint8)

            images = gen_images[0, 0].cpu().numpy()

            masks = real_masks.cpu().numpy()

            # gimg = gen_imgs[0, 0].cpu().numpy()

            bboxs = xywh2x0y0x1y1(bboxs)
            new_bboxs = real_bbox.clone()
            new_bboxs[:, :, 1:] = (bboxs[:, :, 1:] * 255)
            new_bboxs = new_bboxs.to(torch.uint8).cpu().numpy()

            for j, box in enumerate(new_bboxs[0]):
                if box[0] != 0:
                    img = np.zeros_like(images)
                    img[box[2]:box[4], box[1]:box[3]] = images[ box[2]:box[4], box[1]:box[3]]
                    PIL.Image.fromarray(img).convert('L').save(os.path.join(out, f'crops/{((i * num + index)*1000+j):07d}.png'))
            #         # cv2.rectangle(images, (box[1], box[2]), (box[3], box[4]), (255, 0, 0), 2)
            PIL.Image.fromarray(images).convert('L').save(os.path.join(out, f'images/{((i * num + index)*1000):07d}.png'))
            PIL.Image.fromarray(gen_masks[0, 0].cpu().numpy()).convert('L').save(os.path.join(out, f'masks/{((i * num + index)*1000):07d}.png'))

if __name__ == "__main__":
    main()