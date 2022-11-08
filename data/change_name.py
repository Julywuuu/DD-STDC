import os
import tqdm
from PIL import Image

gt_path = '/media/wlj/soft_D/WLJ/WJJ/dataset/camera_4/gtFine/val/headbroken'
#img_path = '/media/wlj/soft_D/WLJ/WJJ/dataset/camera_4/leftImg8bit/val/shoulderbroken'

gt_suffix = '_gtFine_labelIds.png'
img_suffix = '_leftImg8bit.png'

gtlist = os.listdir(gt_path)
for gt in tqdm.tqdm(gtlist):
    newname = gt[:-4] + gt_suffix
    src = os.path.join(gt_path, gt)
    dst = os.path.join(gt_path, newname)
    os.rename(src, dst)

    #newname = gt[:-4] + img_suffix
    #src = os.path.join(img_path, gt)
    #dst = os.path.join(img_path, newname)
    #os.rename(src, dst)