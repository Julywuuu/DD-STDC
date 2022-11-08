#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from models.model_stages_simplify import BiSeNet
from cityscapes_simplify import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
import cv2


class MscEvalV0(object):

    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, dl, n_classes):
        ## evaluate
        hist = torch.zeros(n_classes, n_classes).cuda().detach()
        # if dist.is_initialized() and dist.get_rank() != 0:
        #     diter = enumerate(dl)
        # else:
        diter = enumerate(tqdm(dl))
        for i, (imgs, label) in diter:          #imgs:[batch_size, 3, 1024, 2048]  label:[batch_size, 1, 1024,2048]

            N, _, H, W = label.shape            # N:batch_size; H:1024; W:2048

            label = label.squeeze(1).cuda()     # [5,1,1024,2048] -> [5,1024,2048]
            size = label.size()[-2:]            # [1024,2048]

            imgs = imgs.cuda()

            N, C, H, W = imgs.size()
            new_hw = [int(H*self.scale), int(W*self.scale)]

            imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)     # [4,3,512,1024]
            logits = net(imgs)[0]                                                       # [4,19,512,1024]

            logits = F.interpolate(logits, size=size,mode='bilinear', align_corners=True)   # [4,19,1024,2048]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)


            # show_pred = preds[0]
            # print(show_pred.shape)
            # show_pred = show_pred.detach().cpu()
            # show_pred = np.array(show_pred,dtype=np.uint8)
            # # show_pred = cv2.resize(show_pred,(512,256),interpolation=cv2.INTER_NEAREST)
            # cv2.imshow('img',show_pred)
            # cv2.waitKey(0)
            # pred_list = set((show_pred.reshape(-1).tolist()))
            # print(pred_list)

            # check label
            # lb = set((np.array(label.reshape(-1).detach().cpu())).tolist())
            # print('label:', lb)
            # # check pred
            # pred_set = set((np.array(preds.reshape(-1).detach().cpu())).tolist())
            # print('pred:', pred_set)


            keep = label != self.ignore_label                   # [1,1024,2048]   keee is all 1

            # check keep
            # print(set(np.array(keep.reshape(-1).detach().cpu()).tolist()))
            # print(label.shape)
            # print(keep.shape)
            # print(label[keep].shape)
            # print(preds[keep].shape)

            # after resize need .long();  error
            hist_ =  torch.bincount(((label[keep] * n_classes).long() + preds[keep]),minlength=n_classes ** 2)
            # print(hist_.shape)
            hist_ = hist_.view(n_classes, n_classes).float()
            hist += hist_

        # hist is  hun xiao ju zhen
        ious = hist.diag() / (hist.sum(dim=0) + hist.sum(dim=1) - hist.diag())
        print(ious)                             #  iou about every classes
        miou = ious.mean()

        return miou.item()

def evaluatev0(respth='./pretrained', dspth='./data', backbone='CatNetSmall', scale=0.75, mode='val' , use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False, use_conv_last=False):
    print('scale', scale)
    print('use_boundary_2', use_boundary_2)
    print('use_boundary_4', use_boundary_4)
    print('use_boundary_8', use_boundary_8)
    print('use_boundary_16', use_boundary_16)
    ## dataset
    batchsize = 1
    n_workers = 4
    dsval = CityScapes(dspth, mode=mode)
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    n_classes = 6
    print("backbone:", backbone)
    net = BiSeNet(backbone=backbone, n_classes=n_classes,
     use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, 
     use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16, 
     use_conv_last=use_conv_last)
    net.load_state_dict(torch.load(respth))
    net.cuda()
    net.eval()
    

    with torch.no_grad():
        single_scale = MscEvalV0(scale=scale)
        mIOU = single_scale(net, dl, 6)
    # logger = logging.getLogger()
    # logger.info('mIOU is: %s\n', mIOU)
    print('mIOU is: ', mIOU)


if __name__ == "__main__":


    evaluatev0('/media/wlj/soft_D/WLJ/WJJ/STDC-Seg/checkpoints/camera_4_crop/batch8_10.18_10000it_simplify_dfconvstage2_gray/model_maxmIOU100.pth',
               dspth='./camera_4_crop',
               backbone='STDCNet1446',
               scale=1,
               mode='val',
               use_boundary_2=False, use_boundary_4=False, use_boundary_8=False, use_boundary_16=False)


   

