import os

import PIL
import cv2
import numpy as np
import torch
from tqdm import tqdm

import dnnlib
import legacy
from torch_utils.aug_util import xywh2x0y0x1y1
from torch_utils.custom_ops import bbox_mask
from training.dataset import ImageFolderDataset
from training.training_loop2 import color_mask
from torch.nn import functional as F

COLOR_MAP = {
    "0": (250, 235, 215),
    "1": (240, 255, 255),
    "2": (245, 245, 245),
    "3": (255, 235, 205),
    "4": (227, 207, 87),
    "5": (255, 153, 18),
    "6": (255, 97, 0),
    "7": (128, 42, 42),
    "8": (138, 54, 15),
    "9": (163, 148, 128),
    "10": (255, 125, 64),
    "11": (240, 230, 140),
    "12": (255, 0, 0),
    "13": (139, 69, 19),
    "14": (34, 139, 34),
    "15": (124, 252, 0),
    "16": (0, 255, 127),
    "17": (160, 32, 240),
    "18": (218, 112, 214),
    "19": (0, 199, 140),
    "20": (255, 0, 255),
}

def main():
    batch_size = 1
    out = "sample_T_dmask7"
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
    diter = iter(dataloader)
    C, H, W = test_set[0][0].shape

    # network_pkl = './results/test4/00000-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl'
    network_pkl = './results/test6/00002-dataset4-auto2-noaug/network-snapshot-001000.pkl'
    # network_pkl = '../res/stylegan_ori1/00000-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl'
    # network_pkl = '../res/stylegan/00005-dataset-auto1-noaug/network-snapshot-000800.pkl'
    # network_pkl = '../res/stylegan/stylegan_init/00100-dataset-auto1-batch8/network-snapshot-000600.pkl'

    # G_kwargs = dnnlib.EasyDict(class_name='training.SimNet.SimGenerator', z_dim=64, w_dim=128, bbox_dim=256, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    # G_kwargs.synthesis_kwargs.channel_base = 32768
    # G_kwargs.synthesis_kwargs.channel_max = 512
    # G_kwargs.synthesis_kwargs.num_fp16_res = 4
    # G_kwargs.synthesis_kwargs.conv_clamp = 256
    # G_kwargs.mapping_kwargs.num_layers = 2
    # common_kwargs = dict(c_dim=0, img_resolution=256, img_channels=1)
    # G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().to(device)

    with dnnlib.util.open_url(network_pkl) as f:
        G_pkl = legacy.load_network_pkl(f)['G_ema'].to(device)

    # G.load_state_dict(G_pkl.state_dict())
    G = G_pkl

    start = 10
    count = 100

    seed = 20
    c = 10

    if not os.path.exists(out):
        os.mkdir(out)
    if not os.path.exists(os.path.join(out, "layout")):
        os.mkdir(os.path.join(out, "layout"))

    if not os.path.exists(os.path.join(out, "crops")):
        os.mkdir(os.path.join(out, "crops"))

    if not os.path.exists(os.path.join(out, "bmask")):
        os.mkdir(os.path.join(out, "bmask"))

    if not os.path.exists(os.path.join(out, "gen_images")):
        os.mkdir(os.path.join(out, "gen_images"))

    if not os.path.exists(os.path.join(out, "gen_masks")):
        os.mkdir(os.path.join(out, "gen_masks"))

    if not os.path.exists(os.path.join(out, "gen_masks_01")):
        os.mkdir(os.path.join(out, "gen_masks_01"))
    if not os.path.exists(os.path.join(out, "gen_mid_masks")):
        os.mkdir(os.path.join(out, "gen_mid_masks"))

    if not os.path.exists(os.path.join(out, "real_images")):
        os.mkdir(os.path.join(out, "real_images"))

    if not os.path.exists(os.path.join(out, "real_masks")):
        os.mkdir(os.path.join(out, "real_masks"))

    for i ,data in tqdm(enumerate(dataloader)):

        real_images, real_masks, bboxs = data
        bboxs = bboxs.to(device)
        # print(bboxs)
        # bmask = bbox_mask(device, bboxs[:, :, 1:], 256, 256)*2 - 1
        # PIL.Image.fromarray(color_mask(bmask)[0]).save(os.path.join(out, f'bmask/{(i * 1000)*1000 :09d}.png'))
        bbox2 = xywh2x0y0x1y1(bboxs)
        # print(bbox2)
        bbox2[:, :, 1:] = (bbox2[:, :, 1:] * 255)
        bbox2 = bbox2.to(torch.uint8).cpu().numpy()
        layout = np.zeros([H, W, 3], dtype=np.uint8)
        for h, box in enumerate(bbox2[0]):
            if box[0] != 0:
                # print((box[1], box[2]), (box[3], box[4]), COLOR_MAP[str(h//len(COLOR_MAP))])
                cv2.rectangle(layout, (box[1], box[2]), (box[3], box[4]), COLOR_MAP[str(h%len(COLOR_MAP))], 2)

        PIL.Image.fromarray(layout).save(os.path.join(out, f'layout/{(i*1000)*1000:09d}.png'))
        PIL.Image.fromarray(real_images[0, 0].cpu().numpy()).convert('L').save(
            os.path.join(out, f'real_images/{(i * 1000)*1000 :09d}.png'))
        PIL.Image.fromarray(real_masks[0, 0].cpu().numpy()).convert('L').save(
            os.path.join(out, f'real_masks/{(i * 1000)*1000 :09d}.png'))
        # bbox_mask_ = bbox_mask(bboxs.device, bboxs[:, :, 1:], 256, 256)
        for j in range(c):
            z = torch.from_numpy(np.random.RandomState(seed+j).randn(batch_size, G.z_dim)).to(device)
            fake_images, _, fake_masks, fake_mid_masks = G(z, bboxs, isTrain=False)

            # print(fake_masks.shape)
            fake_images = (fake_images * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            fake_masks1 = (fake_masks.sign() * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            fake_masks2 = F.interpolate(fake_mid_masks*0.5+0.5, (256, 256), mode="bilinear", align_corners=True) * (fake_masks.sign()+1) - 1
            # fake_mid_masks = (torch.sum(F.adaptive_avg_pool2d(fake_mid_masks, (256, 256))*0.5+0.5, dim=1, keepdim=True).clamp(0, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            # fake_mid_masks = ((fake_masks2*0.5+0.5).sum(1, keepdim=True).clamp(0, 1)*255).to(torch.uint8).cpu().numpy()
            fake_masks2 = (color_mask(fake_masks2))
            # fake_masks = (fake_masks.sign() * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            # fake_mid_masks = color_mask(F.interpolate(fake_mid_masks*0.5+0.5, (256, 256), mode="bilinear", align_corners=True))
            fake_mid_masks = (color_mask(F.adaptive_avg_pool2d(fake_mid_masks, (256, 256))))
            # print(fake_masks.shape)
            PIL.Image.fromarray(fake_images[0, 0]).convert('L').save(os.path.join(out, f'gen_images/{(i*1000+j)*1000:09d}.png'))
            PIL.Image.fromarray(fake_masks1[0, 0]).save(os.path.join(out, f'gen_masks_01/{(i*1000+j)*1000:09d}.png'))
            PIL.Image.fromarray(fake_masks2[0]).save(os.path.join(out, f'gen_masks/{(i*1000+j)*1000:09d}.png'))
            PIL.Image.fromarray(fake_mid_masks[0]).save(os.path.join(out, f'gen_mid_masks/{(i*1000+j)*1000:09d}.png'))

            for k, box in enumerate(bbox2[0]):
                if box[0] != 0:
                    img = np.zeros_like(fake_images[0, 0])
                    img[box[2]:box[4], box[1]:box[3]] = fake_images[0, 0, box[2]:box[4], box[1]:box[3]]
                    PIL.Image.fromarray(img).convert('L').save(os.path.join(out, f'crops/{((i * 1000 + j)*1000+k):09d}.png'))


if __name__ == "__main__":
    main()