import glob
import os
import re
import time
from pathlib import Path

import PIL

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


if __name__ == "__main__":
    batch_size = 8
    batch_gpu = 16
    gh, gw = 4, 4
    seed = 20
    row_seeds = 6
    device = torch.device('cuda')
    device2 = torch.device('cpu')

    training_set = ImageFolderDataset(path='../data/dataset3', use_labels=False, bbox_dim=128, max_size=None, xflip=False, aug=False)
    dataloader = iter(torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, pin_memory=True, num_workers=3, prefetch_factor=2))

    C, H, W = training_set[0][0].shape

    # canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    canvas = PIL.Image.new('L', (W * 9, H * batch_size), 'black')
    canvas_s = PIL.Image.new('L', (256 * 5, 256 * batch_size), 'black')
    for _ in range(233):
        next(dataloader)
    images, masks, bbox = next(dataloader)

    imgs = images.to(device).to(torch.float32) / 127.5 - 1
    bboxs = bbox.to(device)

    network_pkl = '../res/stylegan/00004-dataset-auto1-noaug/network-snapshot-001000.pkl'
    #network_pkl = '../res/stylegan/00005-dataset-auto1-noaug/network-snapshot-000800.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)


    z1 = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)
    z2 = torch.from_numpy(np.random.RandomState(seed+10).randn(batch_size, G.z_dim)).to(device)
    z3 = torch.from_numpy(np.random.RandomState(seed+37).randn(batch_size, G.z_dim)).to(device)

    gen_imgs1, gen_masks1, _, _, _ = G(z1, bboxs)
    gen_imgs2, gen_masks2, _, _, _ = G(z2, bboxs)
    gen_imgs3, gen_masks3, _, _, _ = G(z3, bboxs)

    gen_imgs1 = (gen_imgs1 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    gen_masks1 = (gen_masks1 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    gen_imgs2 = (gen_imgs2 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    gen_masks2 = (gen_masks2 * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    images, masks = images.cpu().numpy(), masks.cpu().numpy()
    bboxs = xywh2x0y0x1y1(bboxs)
    bboxs = (bboxs * 256).to(torch.uint8).cpu().numpy()
    for idx, (g_img1, g_img2, g_mask1, g_mask2, img, msk, bbx) in enumerate(zip(gen_imgs1, gen_imgs2, gen_masks1, gen_masks2, images, masks, bboxs)):
        # print(g_mask.shape, g_img.shape, img.shape, msk.shape)
        canvas.paste(PIL.Image.fromarray(msk[0]).convert('L'), (W*0, H*idx))
        canvas.paste(PIL.Image.fromarray(g_mask1[0]).convert('L'), (W*1, H*idx))
        canvas.paste(PIL.Image.fromarray(g_mask2[0]).convert('L'), (W*2, H*idx))
        canvas.paste(PIL.Image.fromarray(img[0]).convert('L'), (W*3, H*idx))
        canvas.paste(PIL.Image.fromarray(g_img1[0]).convert('L'), (W*4, H*idx))
        canvas.paste(PIL.Image.fromarray(g_img2[0]).convert('L'), (W*5, H*idx))
        bbx = bbx[((bbx[:, 2]>0) * (bbx[:, 3]>0))]
        for lab in bbx:
        #lab = bbx[2]
            # print(lab)
            cv2.rectangle(img[0], (lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))
            cv2.rectangle(g_img1[0],(lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))
            cv2.rectangle(g_img2[0],(lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))

        canvas.paste(PIL.Image.fromarray(img[0]).convert("L"), (W*6, H*idx))
        canvas.paste(PIL.Image.fromarray(g_img1[0]).convert("L"), (W*7, H*idx))
        canvas.paste(PIL.Image.fromarray(g_img2[0]).convert("L"), (W*8, H*idx))
    canvas.save(f'./grid_z2.png')