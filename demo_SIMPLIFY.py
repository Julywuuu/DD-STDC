#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from models.model_stages_simplify import BiSeNet
from cityscapes import CityScapes
from PIL import Image
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
from torchvision import transforms

trans = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # transforms.Normalize((0.512, 0.512, 0.512), (0.227, 0.227, 0.227)),
                ])

def initialize_net(save_path):

    backbone = 'STDCNet1446'
    use_boundary_2 = False
    use_boundary_4 = False
    use_boundary_8 = False
    use_boundary_16 = False
    use_conv_last = False
    n_classes = 6

    model = BiSeNet(backbone=backbone, n_classes=n_classes,
         use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4,
         use_boundary_8=use_boundary_8, use_boundary_16=use_boundary_16,
         use_conv_last=use_conv_last)

    # initial model
    model.load_state_dict(torch.load(save_pth))
    model = model.cuda()
    model.eval()

    return model

def file_inference(imgs_path,labels_path,model,img_savepth):
    if not os.path.exists(img_savepth):
        os.mkdir(img_savepth)
    img_names = os.listdir(imgs_path)
    label_names = os.listdir(labels_path)

    for i,(im,lb) in enumerate(zip(img_names, label_names)):
        im_pth = os.path.join(imgs_path, im)
        lb_pth = os.path.join(labels_path,lb)
        single_inference(im_pth,lb_pth,model,img_savepth)

def single_inference(img_path, label_path,model,img_savepth):


    # read img
    img = cv2.imread(img_path,0)

    yuan_img = img.copy()              # opy 1

    # show img
    # img = cv2.resize(img, (768, 576))
    # cv2.imshow('img', img)


    # transform img
    imgs = trans(img)
    imgs = imgs.unsqueeze(0)
    imgs = imgs.cuda()

    # inference
    torch.cuda.synchronize()
    start = time.time()

    logits = model(imgs)[0]                 # out: (out,out16,out32)
    probs = torch.softmax(logits, dim=1)    # (1,19,1024,2048)
    preds = torch.argmax(probs, dim=1)      # (1,1024,2048)

    torch.cuda.synchronize()
    fps = 1 / (time.time() - start)
    print(f'FPS is {fps}')


    # 在原图上把 预测出来的地方标出来
    pred = preds.permute((1, 2, 0)).detach().cpu()  # (1, 1024, 2048) -> (1024, 2048, 1)
    pred = np.array(pred, dtype=np.uint8)  # 转成数组 可以改变像素再用CV2显示
    for row in range(pred.shape[0]):
        for conlumn in range(pred.shape[1]):
            if pred[row][conlumn] in [1, 2, 3, 4, 5]:
                #print(row, ' ', conlumn, ':', pred[row][conlumn])
                #print(img.shape)
                img[row][conlumn] = 255 - pred[row][conlumn] + 1
    # cv2.imshow('pred_img',img)



    ## show result
    show_pred = preds.squeeze(0)  # (1024,2048)
    show_pred = show_pred.detach().cpu()
    show_pred = np.array(show_pred, dtype=np.uint8)
    #show_pred = cv2.resize(show_pred, (768, 576), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow('pred', show_pred * 40)



    ## show label
    label = cv2.imread(label_path, 0)
    #label = cv2.resize(label, (768, 576))
    # cv2.imshow('label', label * 40)



    # # check pred piex
    # pred_list = set((show_pred.reshape(-1).tolist()))
    # print(pred_list)
    #
    # cv2.waitKey(0)


    ## concat 4 imgs
    # yuan_gray = cv2.cvtColor(yuan_img, cv2.COLOR_BGR2GRAY)          # tong yi shape (768,576)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_1 = np.concatenate([yuan_img,img], axis=1)              # heng
    image_2 = np.concatenate([label * 40, show_pred * 40 ], axis=1)
    image = np.concatenate([image_1,image_2], axis=0)                   # zong

    img_name = img_path.split('/')[-1]
    # cv2.imshow(img_name, image)
    cv2.imwrite(os.path.join(img_savepth,img_name),image)
    # cv2.waitKey(0)



if __name__ == '__main__':
    # initial net
    save_pth = '/media/wlj/soft_D/WLJ/WJJ/STDC-Seg/checkpoints/camera_4_crop/batch8_10.18_10000it_simplify_dfconvstage2_gray/model_maxmIOU100.pth'
    model = initialize_net(save_pth)

    ## inference file
    imgs_path = "camera_4_crop/leftImg8bit/test/headround"
    labels_path = "camera_4_crop/gtFine/test/headround"
    img_savepth = 'result/batch8_10.18_10000it_simplify_dfconvstage2_gray+model_maxmIOU100'
    file_inference(imgs_path,labels_path,model,img_savepth)

    ## inference single
    # img_path  = 'camera_4_good.png'
    # label_path = 'camera_4_good_label.png'
    # single_inference(img_path,label_path,model)