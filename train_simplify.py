#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np

from logger import setup_logger
from models.model_stages_simplify import BiSeNet
from cityscapes_simplify import CityScapes
from loss.loss import OhemCELoss
from loss.detail_loss import DetailAggregateLoss
from evaluation import MscEvalV0
from optimizer_loss import Optimizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--local_rank',dest = 'local_rank', type = int,default = 0, )
    parse.add_argument('--n_classes', dest='n_classes', type=int, default=6, )
    parse.add_argument( '--n_workers_train',dest = 'n_workers_train',type = int,default = 8,)
    parse.add_argument('--n_workers_val', dest = 'n_workers_val',type = int,default = 1,)
    parse.add_argument( '--n_img_per_gpu',dest = 'n_img_per_gpu',type = int,default = 8, )      # batchsize
    parse.add_argument('--max_iter',dest = 'max_iter', type = int,default = 10000,)
    parse.add_argument( '--save_iter_sep',dest = 'save_iter_sep', type = int, default = 1000,)
    parse.add_argument( '--warmup_steps', dest = 'warmup_steps', type = int, default = 1000, )
    parse.add_argument( '--mode', dest = 'mode', type = str, default = 'train',  )
    parse.add_argument(  '--ckpt',   dest = 'ckpt',  type = str, default = None, )
    parse.add_argument('--datasetpath', dest='datasetpath', type=str, default='./camera_4_crop', )
    parse.add_argument( '--respath',   dest = 'respath',type = str, default = './checkpoints/camera_4_crop/batch8_10.18_10000it_simplify_dfconvstage1_gray',)
    parse.add_argument( '--backbone', dest = 'backbone',type = str,default = 'STDCNet1446',)
    parse.add_argument('--pretrain_path', dest = 'pretrain_path',type = str,default = '', )
    parse.add_argument( '--use_conv_last', dest = 'use_conv_last', type = str2bool, default = False, )
    parse.add_argument( '--use_boundary_2', dest = 'use_boundary_2', type = str2bool, default = False,)
    parse.add_argument( '--use_boundary_4',dest = 'use_boundary_4', type = str2bool, default = False, )
    parse.add_argument( '--use_boundary_8', dest = 'use_boundary_8', type = str2bool,default = False, )
    parse.add_argument( '--use_boundary_16',dest = 'use_boundary_16', type = str2bool, default = False, )
    return parse.parse_args()

def draw_loss(train_, its):
    # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
    plt.clf()
    x1 = its
    y1 = train_

    # y2 = val_

    plt.title('Train loss', fontsize=20)

    plt.plot(x1, y1, '.-')


    plt.xlabel('it', fontsize=10)
    plt.ylabel('loss', fontsize=20)

    # plt.legend(['train loss', 'val loss'])
    plt.legend(['train loss'])
    plt.grid()                              # 显示网格线
    plt.pause(0.1)


    if its[its.__len__()-1] == parse_args().max_iter-1:
        lossimg_path = os.path.join(parse_args().respath,'train_loss.png')
        plt.savefig(lossimg_path)
        # plt.show()

def train():

    args = parse_args()
    check = 0
    save_pth_path = args.respath
    dataset_path = args.datasetpath
    n_classes = args.n_classes
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2
    mode = args.mode

    cropsize = [448, 704]
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)
    setup_logger(args.respath)

    torch.cuda.set_device(0)

    dataset = CityScapes(dataset_path, cropsize=cropsize, mode=mode, randomscale=randomscale)
    dataloader = DataLoader(dataset,
                    batch_size = n_img_per_gpu,
                    shuffle = True,
                    #sampler = sampler,
                    num_workers = n_workers_train,
                    pin_memory = False,
                    drop_last = True)

    dataset_val = CityScapes(dataset_path, mode='val', randomscale=randomscale)
    dataloader_val = DataLoader(dataset_val,
                    batch_size = 1,
                    shuffle = False,
                    #sampler = sampler_val,
                    num_workers = n_workers_val,
                    drop_last = False)

    ## model
    ignore_idx = 255
    net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, 
    use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8, 
    use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)

    if not args.ckpt is None:
        net.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
        print('load ckpt done')
    net.cuda()
    net.train()

    score_thres = 0.7                                           # 概率小于tresh的piexl才会参与计算损失
    n_min = n_img_per_gpu*cropsize[0]*cropsize[1]//6
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    boundary_loss_func = DetailAggregateLoss()
    ## optimizer
    maxmIOU50 = 0.
    maxmIOU75 = 0.
    maxmIOU100 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5

    ## warm_up : lr form warmup_start_lr imporve to lr_start in warmup_steps
    ##           then reduce from lr_start to zero in (max_iter-warmup_steps)
    ##           poly : lr = base_lr * (1-(epoch/max_epoches))**power

    optim = Optimizer(
            #model = net.module,
            model=net,
            loss = boundary_loss_func,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    
    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    st = glob_st = time.time()
    diter = iter(dataloader)
    epoch = 0

    loss_draw_y = []
    loss_draw_x = []

    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0]==n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dataloader)
            im, lb = next(diter)


        if (check):
            print(im.shape)
            print(lb.shape)

            im_ = np.array(im[0].permute(1, 2, 0))
            lb_ = np.array(np.transpose(lb[0], (1, 2, 0)), dtype=np.uint8)

            # im = cv2.resize(im,(512,384))
            # lb = cv2.resize(lb,(512,384),interpolation=cv2.INTER_NEAREST)

            lb_set = set(lb_.reshape(-1).tolist())
            print(lb_set)

            cv2.imshow('img', im_)
            cv2.imshow('lb', lb_ * 20)

            cv2.waitKey(0)

        # im + lb  start
        im = im.cuda()
        lb = lb.cuda()

        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)



        optim.zero_grad()


        if use_boundary_2 and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail2, detail4, detail8 = net(im)

        if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail4, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
            out, out16, out32, detail8 = net(im)

        if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
            out, out16 = net(im)

        lossp = criteria_p(out, lb.long())
        loss2 = criteria_16(out16, lb.long())
        # loss3 = criteria_32(out32, lb.long())
        
        boundery_bce_loss = 0.
        boundery_dice_loss = 0.
        
        
        # if use_boundary_2:
        #     boundery_bce_loss2,  boundery_dice_loss2 = boundary_loss_func(detail2, lb)
        #     boundery_bce_loss += boundery_bce_loss2
        #     boundery_dice_loss += boundery_dice_loss2
        #
        # if use_boundary_4:
        #     boundery_bce_loss4,  boundery_dice_loss4 = boundary_loss_func(detail4, lb)
        #     boundery_bce_loss += boundery_bce_loss4
        #     boundery_dice_loss += boundery_dice_loss4
        #
        # if use_boundary_8:
        #     boundery_bce_loss8,  boundery_dice_loss8 = boundary_loss_func(detail8, lb)
        #     boundery_bce_loss += boundery_bce_loss8
        #     boundery_dice_loss += boundery_dice_loss8

        loss = lossp + loss2 + boundery_bce_loss + boundery_dice_loss

        loss.backward()
        optim.step()

        loss_avg.append(loss.item())

        # loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_bce.append(boundery_bce_loss)         # 0
        # loss_boundery_dice.append(boundery_dice_loss.item())
        loss_boundery_dice.append(boundery_dice_loss)          # 0


        ## print training log message
        if (it+1)%msg_iter==0:          # 50 it print
            loss_avg = sum(loss_avg) / len(loss_avg)

            # draw
            loss_draw_y.append(loss_avg)
            loss_draw_x.append(it)
            draw_loss(loss_draw_y, loss_draw_x)

            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            # loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            # loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                # 'boundery_bce_loss: {boundery_bce_loss:.4f}',
                # 'boundery_dice_loss: {boundery_dice_loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it = it+1,
                max_it = max_iter,
                lr = lr,
                loss = loss_avg,
                # boundery_bce_loss = loss_boundery_bce_avg,
                # boundery_dice_loss = loss_boundery_dice_avg,
                time = t_intv,
                eta = eta
            )
            
            logger.info(msg)
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            st = ed

        # eval net
        if (it+1)%save_iter_sep==0:# and it != 0:

            logger.info('evaluating the model ...')
            logger.info('setup and restore model')
            
            net.eval()

            logger.info('compute the mIOU')
            with torch.no_grad():
                single_scale1 = MscEvalV0()
                mIOU50 = single_scale1(net, dataloader_val, n_classes)           # caculate MIOU50

                single_scale2= MscEvalV0(scale=0.75)
                mIOU75 = single_scale2(net, dataloader_val, n_classes)           ## caculate MIOU75

                single_scale3 = MscEvalV0(scale=1)
                mIOU100 = single_scale3(net, dataloader_val, n_classes)

            save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}_mIOU100_{}.pth'
            .format(it+1, str(round(mIOU50,4)), str(round(mIOU75,4)), str(round(mIOU100,4))))                      # ound() save 4wei xiaoshu
            
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            #if dist.get_rank()==0:
            torch.save(state, save_pth)

            logger.info('training iteration {}, model saved to: {}'.format(it+1, save_pth))

            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
                save_pth = osp.join(save_pth_path, 'model_maxmIOU50.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                #if dist.get_rank()==0:
                torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))
            
            if mIOU75 > maxmIOU75:
                maxmIOU75 = mIOU75
                save_pth = osp.join(save_pth_path, 'model_maxmIOU75.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                # if dist.get_rank()==0:
                torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))

            if mIOU100 > maxmIOU100:
                maxmIOU100 = mIOU100
                save_pth = osp.join(save_pth_path, 'model_maxmIOU100.pth'.format(it+1))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                # if dist.get_rank()==0:
                torch.save(state, save_pth)
                logger.info('max mIOU model saved to: {}'.format(save_pth))
            
            logger.info('mIOU50 is: {}, mIOU75 is: {}, mIOU100 is: {}'.format(mIOU50, mIOU75, mIOU100))
            logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}, maxmIOU100 is: {}.'.format(maxmIOU50, maxmIOU75,maxmIOU100))

            net.train()
    
    ## dump the final model
    save_pth = osp.join(save_pth_path, 'model_final.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    #if dist.get_rank()==0:
    torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)

if __name__ == "__main__":
    train()
