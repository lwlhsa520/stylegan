# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
import importlib
import hashlib
import shutil
from pathlib import Path

from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.
from torchvision import transforms

from torch_utils.aug_util import xywh2x0y0x1y1

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)

    try: # pylint: disable=too-many-nested-blocks
        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        # Compile and load.
        verbose_build = (verbosity == 'full')

        # Incremental build md5sum trickery.  Copies all the input source files
        # into a cached build directory under a combined md5 digest of the input
        # source files.  Copying is done only if the combined digest has changed.
        # This keeps input file timestamps and filenames the same as in previous
        # extension builds, allowing for fast incremental rebuilds.
        #
        # This optimization is done only in case all the source files reside in
        # a single directory (just for simplicity) and if the TORCH_EXTENSIONS_DIR
        # environment variable is set (we take this as a signal that the user
        # actually cares about this.)
        source_dirs_set = set(os.path.dirname(source) for source in sources)
        if len(source_dirs_set) == 1 and ('TORCH_EXTENSIONS_DIR' in os.environ):
            all_source_files = sorted(list(x for x in Path(list(source_dirs_set)[0]).iterdir() if x.is_file()))

            # Compute a combined hash digest for all source files in the same
            # custom op directory (usually .cu, .cpp, .py and .h files).
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())
            build_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build) # pylint: disable=protected-access
            digest_build_dir = os.path.join(build_dir, hash_md5.hexdigest())

            if not os.path.isdir(digest_build_dir):
                os.makedirs(digest_build_dir, exist_ok=True)
                baton = FileBaton(os.path.join(digest_build_dir, 'lock'))
                if baton.try_acquire():
                    try:
                        for src in all_source_files:
                            shutil.copyfile(src, os.path.join(digest_build_dir, os.path.basename(src)))
                    finally:
                        baton.release()
                else:
                    # Someone else is copying source files under the digest dir,
                    # wait until done and continue.
                    baton.wait()
            digest_sources = [os.path.join(digest_build_dir, os.path.basename(x)) for x in sources]
            torch.utils.cpp_extension.load(name=module_name, build_directory=build_dir,
                verbose=verbose_build, sources=digest_sources, **build_kwargs)
        else:
            torch.utils.cpp_extension.load(name=module_name, verbose=verbose_build, sources=sources, **build_kwargs)
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------

def bbox_mask(device, bbox, H, W):
    b, o, _ = bbox.size()
    N = b * o

    bbox_1 = bbox.float().view(-1, 4) # x, y, w, h
    ww, hh = bbox_1[:, 2], bbox_1[:, 3]
    x0, y0 = bbox_1[:, 0] - ww / 2, bbox_1[:, 1] - hh/ 2,
    # x0, y0 = bbox_1[:, 0], bbox_1[:, 1]

    x0 = x0.contiguous().view(N, 1).expand(N, H)
    ww = ww.contiguous().view(N, 1).expand(N, H)
    y0 = y0.contiguous().view(N, 1).expand(N, W)
    hh = hh.contiguous().view(N, 1).expand(N, W)

    X = torch.linspace(0, 1, steps=W).view(1, W).expand(N, W).to(device=device)
    Y = torch.linspace(0, 1, steps=H).view(1, H).expand(N, H).to(device=device)

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    X_out_mask = ((X < 0) + (X > 1)).view(N, 1, W).expand(N, H, W)
    Y_out_mask = ((Y < 0) + (Y > 1)).view(N, H, 1).expand(N, H, W)

    out_mask = 1 - (X_out_mask + Y_out_mask).float().clamp(max=1.0)
    return out_mask.view(b, o, H, W)




def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    ww, hh = boxes[:, 2], boxes[:, 3]
    x0, y0 = boxes[:, 0] - ww/2, boxes[:, 1] - hh/2

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid

def masks_to_layout(boxes, masks, H, W=None):
    """
    Inputs:
        - boxes: Tensor of shape (b, num_o, 4) giving bounding boxes in the format
            [x0, y0, x1, y1] in the [0, 1] coordinate space
        - masks: Tensor of shape (b, num_o, M, M) giving binary masks for each object
        - H, W: Size of the output image.
    Returns:
        - out: Tensor of shape (N, num_o, H, W)
    """
    b, num_o, _ = boxes.size()
    M = masks.size(2)
    assert masks.size() == (b, num_o, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes.view(b * num_o, -1), H, W).float().to(device=masks.device)

    img_in = masks.float().view(b*num_o, 1, M, M)
    sampled = F.grid_sample(img_in, grid, mode='bilinear')

    return sampled.view(b, num_o, H, W)

#----------------------------------------------------------------------------

def mask2bbox(mask):
    # i = 16
    # mask = cv2.imread(f"./post_mask_{i}.png")
    # mask = torch.from_numpy(mask.transpose(2, 0, 1)).to(torch.uint8)  # c, h, w
    # param mask:  value [-1, 1] tensor[resolution, resolution]
    #
    resolution = mask.shape[-1]
    mask = (mask * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    h, w = torch.where(mask > 236)
    res = []
    if len(h):
        xmin, xmax, ymin, ymax = h.min(), h.max(), w.min(), w.max(),
        # print(f"({xmin}, {ymin}), ({xmax}, {ymax})")
        mask = mask[xmin:xmax, ymin:ymax]
        # transforms.ToPILImage()(mask).save("img.png")

        x3, y3 = (xmax - xmin), (ymax - ymin)
        x0, y0 = 0, 0
        x, y = torch.where(mask <= 236)
        yvalue, xvalue = y[x == 0], x[y == 0]

        if len(xvalue) or len(yvalue):
            y1 = yvalue.min() if len(yvalue) else y3  # 1-1 up_right
            y2 = yvalue.max() if len(yvalue) else 0  # 1-2 up_left
            x1 = xvalue.min() if len(xvalue) else x3  # 1_1 left_bottom
            x2 = xvalue.max() if len(xvalue) else 0  # 2_1 up_left
            # h :x0 < x1 < x2 < x3  w: y0 < y1 < y2 < y3
            y0, y1, y2, y3, x0, x1, x2, x3 = ymin+y0, ymin+y1, ymin+y2, ymin+y3, xmin+x0, xmin+x1, xmin+x2, xmin+x3
            up_left, up_right, left_bottom, right_bottom = [y0, x0, y1-y0, x1-x0], [y2, x0, y3-y2, x1-x0], [y0, x2, y1-y0, x3-x2], [y2, x2, y3-y2, x3-x2]

            if (up_left[2] * up_left[3]) >0:
                res.append(up_left)
            if up_right not in res and (up_right[2] * up_right[3]>0):
                res.append(up_right)
            if left_bottom not in res and (left_bottom[2] * left_bottom[3])>0:
                res.append(left_bottom)
            if right_bottom not in res and (right_bottom[2] * right_bottom[3])>0:
                res.append(right_bottom)

        else:
            res.append([ymin, xmin, ymax-ymin, xmax-xmin] if (ymax-ymin)*(xmax-xmin)>0 else [] )
    return torch.Tensor(res)

def batch_Mask2bbox(bmasks, res = 256, resample=8):
    aug_bbox = []
    for batch, bmask in enumerate(bmasks):
        aug_s_bbox = torch.empty([0, 4])
        count = 0
        for bm in bmask:
            bbs = mask2bbox(bm)
            # print("pre:", bbs)
            if bbs.ndim == 2:
                bbs = bbs[bbs[:, 2] > 0]
                bbs = bbs[bbs[:, 3] > 0]
            # print("post:", bbs)

            if len(bbs):
                aug_s_bbox = torch.cat([aug_s_bbox, bbs])
                count += (len(bbs) - 1)
            else:
                count -= 1
        num = len(aug_s_bbox)
        ids = np.random.randint(0, num, resample)
        aug_s_bbox = aug_s_bbox[ids % num]
        aug_bbox.append(aug_s_bbox)
    bbox = torch.stack(aug_bbox)
    return bbox

#-------------------------------------------------------------------------------------

def img_resampler(img, bbox, resample_num=16, real_img=None, imgs_size=32):
    B, C, H, W = img.shape
    bbox = bbox * (W-1) # x, y, w, h
    bi, ni = torch.where((bbox[:, :, 2] * bbox[:, :, 3])> 0)
    bc = np.array([len(bi[bi == b]) for b in range(img.shape[0])]) # batch count
    rs = torch.cat([torch.randint(bc[:i].sum(), bc[:i+1].sum(), [resample_num]) for i in range(bbox.shape[0])]) # resample

    bbox2 = bbox[bi[rs], ni[rs]]
    bbox2 = xywh2x0y0x1y1(bbox2)
    bbox2 = bbox2.clamp(0, 255).to(torch.uint8)

    ims = torch.zeros([B * resample_num, C, imgs_size, imgs_size], device=img.device)
    r_ims = torch.zeros([B * resample_num, C, imgs_size, imgs_size], device=img.device)

    transform = transforms.Resize([imgs_size, imgs_size])

    for idx, (im, box) in enumerate(zip(img, bbox2)):
        ims[idx] = transform(im[:, box[0]:box[2], box[1]:box[3]])

    if real_img is not None:
        for idx, (r_im, box) in enumerate(zip(real_img, bbox2)):
            r_ims[idx] = transform(r_im[:, box[0]:box[2], box[1]:box[3]])

        return ims, r_ims
    # return ims.view(B, resample_num, C, imgs_size, imgs_size)

    return ims
