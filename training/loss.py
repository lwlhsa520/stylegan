# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.custom_ops import img_resampler
from torch_utils.ops import conv2d_gradfix
import torch.nn.functional as F

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G,  Ds, Dm, P, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G = G
        self.Ds = Ds
        self.Dm = Dm
        self.P = P
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

    def run_G(self, z, bbox, sync):
        with misc.ddp_sync(self.G, sync):
            img, mask, ws, ws2, mid_mask = self.G(z, bbox)
        return img, mask, ws, ws2, mid_mask

    # def run_Dsr(self, img, sync):
    #     with misc.ddp_sync(self.Dsr, sync):
    #         img = self.Dsr(img)
    #
    #     return img

    def run_D(self, img, mask, bbox, sync):

        with misc.ddp_sync(self.Ds, sync):
            sample = img_resampler(img, bbox, resample_num=16) # (B*16, C, H, W)
            d2 = self.Ds(sample).view(img.shape[0], 16, 1).mean(1)

        if self.augment_pipe is not None:
            img, mask = self.augment_pipe(img, mask)

        with misc.ddp_sync(self.Dm, sync):
            d1 = self.Dm(img, mask)

        return d1, d2

    def accumulate_gradients(self, phase, real_img, real_mask, real_bbox, gen_z, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, gen_mask, _gen_ws, gen_ws2, gen_mid_mask = self.run_G(gen_z, real_bbox, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                # gen_img = self.run_Dsr(gen_img, sync=False)
                gen_logits, gen_logits2 = self.run_D(gen_img, gen_mask, real_bbox, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                loss_Gmain = loss_Gmain * (1 - gen_logits2)
                training_stats.report('Loss/G/loss', loss_Gmain)

                # recon_loss = self.P(gen_img, real_img) + self.P(gen_mask, real_mask)
                recon_loss = self.P(real_img, gen_img) + self.P(gen_mid_mask, F.adaptive_avg_pool2d(real_mask, gen_mid_mask.shape[2:4])) + self.P(gen_mask, real_mask)
                training_stats.report('Loss/reconloss', recon_loss)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain + recon_loss).mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_mask, gen_ws, gen_ws2, _gen_mid_mask = self. run_G(gen_z[:batch_size], real_bbox[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                pl_noise2 = torch.randn_like(gen_mask) / np.sqrt(gen_mask.shape[2] * gen_mask.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl1, pl2 = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum(), (gen_mask * pl_noise2).sum()], inputs=[gen_ws, gen_ws2], create_graph=True, only_inputs=True)[:2]
                pl_lengths = (pl1.square().sum(2).mean(1) + pl2.square().sum(2).mean(1)).sqrt()
                # pl_lengths = pl1.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + gen_mask[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, gen_mask, _gen_ws, _gen_ws2, _gen_mid_mask = self.run_G(gen_z, real_bbox, sync=False)
                # gen_img = self.run_Dsr(gen_img, sync=False)
                gen_logits, gen_logits2 = self.run_D(gen_img, gen_mask, real_bbox, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                loss_Dgen = loss_Dgen * (1 + gen_logits2)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # resolution = real_img.shape[2] // 2
                # real_img_coase = F.adaptive_avg_pool2d(real_img, (resolution, resolution))
                # real_img_coase2 = F.adaptive_avg_pool2d(real_img_coase, real_img.shape[2:4])
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_mask_tmp = real_mask.detach().requires_grad_(do_Dr1)
                # real_bbox_tmp = real_bbox.detach().requires_grad_(do_Dr1)

                sr_loss = 0
                # sr_img= self.run_Dsr(real_img_coase2, sync=sync)
                # sr_loss = torch.nn.functional.l1_loss(sr_img, real_img) + self.P(sr_img, real_img)

                real_logits, real_logits2 = self.run_D(real_img_tmp, real_mask_tmp, real_bbox, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal = loss_Dreal * (1 - real_logits2)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        # r1_img, r1_mask = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp, real_mask_tmp], create_graph=True, only_inputs=True)[:2]
                        r1_img = torch.autograd.grad(outputs=[real_logits.sum(), real_logits2.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        # r1_img = torch.autograd.grad(outputs=[real_logits2.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_img.square().sum([1,2,3])
                    r1_penalty1 = 0
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2) + r1_penalty1 * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/r1_penalty1', r1_penalty1)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + real_logits2 * 0 + loss_Dreal + sr_loss + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
