#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *
import cv2
import torch.nn.functional as F


class CityScapes(Dataset):
    def __init__(self, rootpth, cropsize=(512, 384), mode='train',
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)                                         #lel_info is list
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}           # el is dict ; lb_map is dict

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)      # img
        folders = os.listdir(impth)                         # class folders

        for fd in folders:
            fdpth = osp.join(impth, fd)                     #
            im_names = os.listdir(fdpth)                    #

            names = [el.replace('_leftImg8bit.png', '') for el in im_names]     # qu diao hou zhui de names

            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)                          # imgnames: all names without _leftImg8bit.png;
            self.imgs.update(dict(zip(names, impths)))      # self_imgs:  {{name:impths},{ },{ }}

        ## parse gt directory   gt is ground truth
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))        # self_labels:  {{name:impths},{ },{ }}    key:value

        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)


        ## check!
        # for i in range(imgnames.__len__()):
        #
        #     if imgnames[i]==gtnames[i]:
        #         continue
        #     else:
        #         print(imgnames[i])
        #         print(gtnames[i])



        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            ])
        self.trans_train = Compose([
            #ColorJitter( brightness = 0.5,contrast = 0.5,saturation = 0.5),        # change color
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            ###RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            ###RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]

        img = Image.open(impth).convert('RGB')
        # img.show()

        # img = cv2.imread(impth, 1)
        label = Image.open(lbpth).convert('L')


        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)


        #label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = np.array(label).astype(np.int64)

        # resize
        #label = cv2.resize(label, (512, 384), interpolation=cv2.INTER_NEAREST)


        label = np.expand_dims(label,0)


        # resize
        # img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        # img = F.interpolate(img, (384, 512), mode='nearest' )
        # img = img.squeeze(0)
        # img = img.float()




        #label = self.convert_labels(label)
        # print(img.shape)
        # print(label.shape)

        # if img.shape[1]==384 and img.shape[2]==512:
        #     print(fn)
        return img, (torch.from_numpy(label))

    def __len__(self):
        return self.len


    # def convert_labels(self, label):
    #     for k, v in self.lb_map.items():        # lb_map: {id:train_id},
    #         label[label == k] = v
    #     return label



if __name__ == "__main__":
    from tqdm import tqdm
    # ds = CityScapes('./data/', n_classes=19, mode='val')
    ds = CityScapes('./camera_4_crop/', mode='train')
    #uni = []
    for i, (im, lb) in enumerate(ds):
        # continue
        #lb_uni = np.unique(lb).tolist()
        #uni.extend(lb_uni)
    # print(uni)
    # print(set(uni))

        print('img:', im.shape,'type:',type(im))
        print('label:',lb.shape,'type:',type(lb))



        im = np.array(im.permute(1,2,0))
        #lb = np.array(lb.transpose(1,2,0),dtype=np.uint8)
        lb = np.array(lb.permute(1,2,0),dtype=np.uint8)
        # im = cv2.resize(im,(512,384))
        # lb = cv2.resize(lb,(512,384),interpolation=cv2.INTER_NEAREST)


        lb_set = set(lb.reshape(-1).tolist())
        print(lb_set)

        cv2.imshow('img',im)
        cv2.imshow('lb',lb*20)

        cv2.waitKey(0)




