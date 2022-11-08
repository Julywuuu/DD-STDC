import numpy as np
import torchvision.transforms
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image
import torch
import sys
import numpy
from tabulate import tabulate

sys.path.insert(0, '.')
import os
import numpy as np
import cv2
import argparse
# from lib.models import model_factory
from models.model_stages import BiSeNet
from configs import set_cfg_from_file


np.set_printoptions(threshold=np.inf,linewidth=300)

transformer = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# labels为你的像素值的类别
def get_miou_recall_precision(label_image, pred_image, labels):
    '''
        混淆矩阵
        Recall、Precision、MIOU计算
        label_image: 标签图路径
        pred_image： 预测图路径
        labels： [0,1,2]类别列表
    '''
    print(label_image.shape)
    print(pred_image.shape)
    label = label_image.reshape(-1)
    lens = len(labels)
    pred = pred_image.reshape(-1)
    out = confusion_matrix(label, pred, labels=labels)  # 拉成向量 再求混淆矩阵
    print(out.shape)
    print(out)
    r, l = out.shape
    iou_temp = 0       # IOU的和
    # recall = {}
    # precision = {}

    # for i in range(1, r):  ## bu dai background
    for i in range(r):
        TP = out[i][i]
        temp = np.concatenate((out[0:i, :], out[i + 1:, :]), axis=0)
        sum_one = np.sum(temp, axis=0)
        FP = sum_one[i]
        temp2 = np.concatenate((out[:, 0:i], out[:, i + 1:]), axis=1)
        FN = np.sum(temp2, axis=1)[i]
        TN = temp2.reshape(-1).sum() - FN

        if FN + TP + FP == 0:
            lens -= 1
            continue

        iou_temp += (TP / (TP + FP + FN))
        # recall[i] = TP / (TP + FN)
        # precision[i] = TP / (TP + FP)

    # MIOU = iou_temp / (lens-1)  # 不带背景的话
    # return MIOU, recall, precision
    MIOU = iou_temp / lens
    return MIOU

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_path', type=str,
                       default=r"./res/cam3_9.6_(512,384)_300.pth", )     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parse.add_argument('--config', dest='config', type=str, default=r'../configs/bisenetv2_city.py', )
    parse.add_argument('--images_path', dest='images_path', type=str,
                       default='../dataset/camera_3/val/img')
    parse.add_argument('--labels_path', dest='labels_path', type=str,
                       default='../dataset/camera_3/val/label')
    return parse.parse_args()


# 单张图片推理,返回predication图和label的 int类型
def inference(img_path, lb_path):
    #net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
    net = BiSeNet(backbone="STDCNet1446", n_classes=4,
                  use_boundary_2=False, use_boundary_4=False,
                  use_boundary_8=True, use_boundary_16=False,
                  use_conv_last=False)
    # net.load_state_dict(torch.load(args.weight_path), strict=False)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
    net.cuda()
    net.eval()
    torch.cuda.empty_cache()

    img = cv2.imread(img_path)[:, :, ::-1]                               # img 需要输入net,所以读入RGB (512,512,3)
    #ret, th = cv2.threshold(img, 145, 255, cv2.THRESH_BINARY)
    img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_NEAREST)

    label = cv2.imread(lb_path, 0)  # lable (512, 512)
    label = cv2.resize(label, (512, 384), interpolation=cv2.INTER_NEAREST)

    # img = torch.Tensor(img).permute((2, 0, 1)).unsqueeze(dim=0).cuda()  # (1, 3, 512, 512) 可以输入网络了
    img = transformer(img).unsqueeze(dim=0).cuda()                        # 小心输入的 是 不是 归一化后的数据

    pred = net(img)
    pred = pred.squeeze(dim=0).detach().cpu()

    return np.array(pred).astype(int), label.astype(int)    # 计算混淆矩阵的函数只接受int类型的array,label读进来就是array


def get_images_miou(images_path, labels_path):
    images_list = os.listdir(images_path)
    labels_list = os.listdir(labels_path)

    assert len(images_list) == len(labels_list)
    HEADS, MIOUS = [], []
    recalls, precisions = [], []
    for i, (img, lb) in enumerate(zip(images_list, labels_list)):   # 同时取到两个list的内容 需要加 zip
        img_path = os.path.join(images_path, img)
        lb_path = os.path.join(labels_path, lb)
        pred, lb = inference(img_path, lb_path)  # 推理
        MIOU = get_miou_recall_precision(lb, pred, labels)          # 求混淆矩阵,在计算MIOU
        MIOUS.append(MIOU)
        HEADS.append(img)
        # recalls.append(recall)
        # precisions.append(precision)
        # print("MIOU:{}\nrecall:{}\nprecison:{}".format(MIOU, recall, precision))
    HEADS.append('MIoU')
    MIOUS.append(sum(MIOUS) / len(MIOUS))
    print(tabulate([MIOUS,], headers=HEADS, tablefmt='orgtbl'))



if __name__ == '__main__':
    args = parse_args()
    cfg = set_cfg_from_file(args.config)
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # ------------------- 单张 ------------------------------
    # pred, lab = inference(args.image, args.label)
    # MIOU = get_miou_recall_precision(lab, pred, labels)
    # print('MIOU:{:.3f}'.format(MIOU))
    # ------------------------------------------------------

    # ---------------------多张------------------------------
    get_images_miou(args.images_path, args.labels_path)
    # ------------------------------------------------------
