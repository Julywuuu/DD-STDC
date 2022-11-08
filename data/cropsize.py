import cv2
import numpy
import os

img_bootpath = '/media/wlj/soft_D/WLJ/WJJ/dataset/camera_4/gtFine/'
save_bootpath = '/media/wlj/soft_D/WLJ/WJJ/dataset/camera_4_crop/gtFine/'
y_start = 500
x_start = 500
h = 448
w = 704

datatypes = os.listdir(img_bootpath)                                        # train val test
for datatype_ in datatypes:
    classes = os.listdir(os.path.join(img_bootpath,datatype_))              # headbroken headdeep nok
    for class_ in classes:
        img_names = os.listdir(os.path.join(img_bootpath,datatype_,class_)) # names
        for img_ in img_names:
            img_path = os.path.join(img_bootpath,datatype_,class_,img_)
            img = cv2.imread(img_path,1)

            #assert img.shape[1]>1000

            ret = img[y_start:y_start + h, x_start:x_start + w].copy()

            save_classpath = os.path.join(save_bootpath,datatype_,class_)
            if not os.path.exists(save_classpath):
                os.makedirs(save_classpath)
            save_path = os.path.join(save_classpath,img_)
            cv2.imwrite(save_path, ret)



