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
    out = "./data2/"
    device = torch.device('cuda')
    device2 = torch.device('cpu')
    # _label = generate_labels(batch_gpu, seed=10)
    # my_label = torch.as_tensor(np.loadtxt("./data/data_0_1.txt"), device=device)
    # if len(my_label)<128:
    #     my_label = torch.cat([my_label, torch.zeros([128-len(my_label), 5], device=device)], 0)
    # print(my_label)
    test_set = ImageFolderDataset(path='../data/dataset', use_labels=False, bbox_dim=128, max_size=None, xflip=False, aug=False)
    dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, pin_memory=True, num_workers=3, prefetch_factor=2)

    C, H, W = test_set[0][0].shape

    network_pkl = '../res/test/00000-dataset4-auto1/network-snapshot-001200.pkl'
    # network_pkl = '../res/stylegan/00005-dataset-auto1-noaug/network-snapshot-000800.pkl'
    # network_pkl = '../res/stylegan/stylegan_init/00100-dataset-auto1-batch8/network-snapshot-000600.pkl'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)


    if not os.path.exists(out):
        os.makedirs(out)
    # with dnnlib.util.open_url(network_pkl) as f:
    #     D = legacy.load_network_pkl(f)['D'].to(device)
    # common_kwargs = dnnlib.EasyDict(channel_base=16384, channel_max=512, num_fp16_res=4, conv_clamp=256)
    # synthesis_kwargs = dnnlib.EasyDict(channel_base=16384//2, channel_max=512, num_fp16_res=4, conv_clamp=256)
    # mapping_kwargs = dnnlib.EasyDict(num_layers = 2)
    # epilogue_kwargs = dnnlib.EasyDict(mbstd_group_size = 4)
    # Gnet = SimGenerator(mapping_kwargs=mapping_kwargs, synthesis_kwargs=common_kwargs).eval().to(device)
    # Gnet.load_state_dict(G.state_dict())
    #
    # Dnet.load_state_dict(D.state_dict())
    #
    # vgg_loss = VGGLoss().eval().to(device)
    # Perceptual_loss = Perceptual_loss134().eval().to(device)

    # canvas = PIL.Image.new('RGB', (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    canvas = PIL.Image.new('L', (W * 6, H * batch_size), 'black')
    canvas_s = PIL.Image.new('L', (256 * 5, 256 * batch_size), 'black')

    for index, data in enumerate(tqdm(dataloader)):
        images, masks, bbox = data
        bboxs = bbox.to(device)
        z = torch.from_numpy(np.random.RandomState(seed).randn(batch_size, G.z_dim)).to(device)
        gen_imgs, gen_masks, _, _, gen_mid_masks = G(z, bboxs)
        gen_imgs = (gen_imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        gen_mid_masks = (F.interpolate(gen_mid_masks, (256, 256), mode='nearest').sum(1, keepdim=True) * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        gen_masks = (gen_masks.sign() * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        bboxs = xywh2x0y0x1y1(bboxs)
        bboxs = (bboxs * 256).to(torch.uint8).cpu().numpy()
        # y = y.to(torch.uint8).cpu().numpy()
        for idx, (g_img, g_m_mask, g_mask, img, msk, bbx) in enumerate(zip(gen_imgs, gen_mid_masks, gen_masks, images, masks, bboxs)):
            # print(g_mask.shape, g_img.shape, img.shape, msk.shape)
            canvas.paste(PIL.Image.fromarray(msk[0]).convert('L'), (W*0, H*idx))
            canvas.paste(PIL.Image.fromarray(g_mask[0]).convert('L'), (W*1, H*idx))
            canvas.paste(PIL.Image.fromarray(img[0]).convert('L'), (W*2, H*idx))
            canvas.paste(PIL.Image.fromarray(g_img[0]).convert('L'), (W*3, H*idx))
            bbx = bbx[((bbx[:, 2]>0) * (bbx[:, 3]>0))]
            for lab in bbx:
            #lab = bbx[2]
                # print(lab)
                cv2.rectangle(img[0], (lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))
                cv2.rectangle(g_img[0],(lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))

            canvas.paste(PIL.Image.fromarray(img[0]).convert("L"), (W*4, H*idx))
            canvas.paste(PIL.Image.fromarray(g_img[0]).convert("L"), (W*5, H*idx))
        canvas.save(f'./{out}/grid{index}.png')


if __name__ == "__main__":
    main()