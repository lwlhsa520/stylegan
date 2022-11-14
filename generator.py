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


class Gen(torch.nn.Module):
    def __init__(self):
        super().__init__()

        batch_size = 8
        start = 0
        length = 100
        self.init_seed = 20
        self.outdir = '../Gen/'

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # if not os.path.exists(self.outdir+'/images'):
        #     os.makedirs(self.outdir+'/images')
        # if not os.path.exists(self.outdir+'/masks'):
        #     os.makedirs(self.outdir+'/masks')

        self.path = '../data/dataset3'
        network_pkl = '../res/stylegan/00003-dataset-auto2-noaug/network-snapshot-001400.pkl'
        self.device_gpu = torch.device('cuda')
        self.device_cpu = torch.device('cpu')
        self.training_set = ImageFolderDataset(path=self.path, use_labels=False, bbox_dim=128, max_size=None,
                                          xflip=False, aug=False)

        self.loadGenMode(network_pkl)

    def getBbox(self):
        bboxs = []
        for idx in range(len(self.training_set)):
            image, mask, bbox = self.training_set[idx]
            bboxs.append(bbox)
        return np.array(bboxs)

    def genSingleImgs(self):
        bboxs = self.getBbox()
        for seed_idx, bbox in enumerate(bboxs):
            bbox = torch.from_numpy(bbox[None, :, :]).to(self.device_gpu)
            print('Generating image for (%d/%d) ...' % (seed_idx, len(self.training_set)))
            z = torch.from_numpy(np.random.RandomState(self.init_seed + seed_idx).randn(1, self.net.z_dim)).to(self.device_gpu)
            img, mask, _, _, _ = self.net(z, bbox)
            canvas = PIL.Image.new('L', (256 * 3, 256 * 1), 'black')
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            mask = (mask * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            canvas.paste(PIL.Image.fromarray(img[0, 0]).convert("L"), (256 * 0, 256 * 0))
            canvas.paste(PIL.Image.fromarray(mask[0, 0]).convert("L"), (256 * 1, 256 * 0))
            # bbox = bbox[((bbox[:, 2] > 0) * (bbox[:, 3] > 0))]
            bbox = xywh2x0y0x1y1(bbox)
            bbox = (bbox * 256).to(torch.uint8).cpu().numpy()
            # print(bbox)
            for lab in bbox[0]:
                # print((lab[0], lab[1]), (lab[2], lab[3]))
                cv2.rectangle(img[0, 0], (lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))
            canvas.paste(PIL.Image.fromarray(img[0, 0]).convert("L"), (256 * 2, 256 * 0))
            canvas.save(f'{self.outdir}/seed{seed_idx:04d}.png')


    def loadGenMode(self, network_pkl):
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device_gpu)
        self.net = G
        return G

    def loadGenModeWithNet(self, network_pkl):
        G = self.loadGenMode(network_pkl)
        common_kwargs = dnnlib.EasyDict(channel_base=16384, channel_max=512, num_fp16_res=4, conv_clamp=256)
        mapping_kwargs = dnnlib.EasyDict(num_layers=2)
        Gnet = SimGenerator(mapping_kwargs=mapping_kwargs, synthesis_kwargs=common_kwargs).eval().to(self.device_gpu)
        Gnet.load_state_dict(G.state_dict())
        self.net = Gnet

if __name__ == "__main__":
    Gen().genSingleImgs()