import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# THETA_RADIUS_GROUP = [0.0, np.pi*1/4, np.pi*2/4, np.pi*3/4] #gabor方向，0°，45°，90°，135°，共四个
# GK_SIZES = [7,9,11,13,15,17]

THETA_RADIUS_GROUP = [0, np.pi*2/4] # gabor方向，0°，45°，90°，135°，共四个
GK_SIZES = [7,11,15]


# 构建Gabor滤波器
def build_filters():
     filters = []
     # ksize = [7,9,11,13,15,17] # gabor尺度，6个
     lamda = np.pi/2.0         # 波长
     for theta in THETA_RADIUS_GROUP:
         for KSIZE in GK_SIZES:
             kern = cv2.getGaborKernel((KSIZE, KSIZE), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
             kern /= 1.5*kern.sum()
             filters.append(kern)
     plt.figure(1)

     # 用于绘制滤波器
     for temp in range(len(filters)):
         plt.subplot(len(THETA_RADIUS_GROUP), len(GK_SIZES), temp + 1)
         plt.imshow(filters[temp])
     plt.show()
     return filters

# Gabor特征提取
def getGabor(img,filters):
    res = [] # 滤波结果
    for i in range(len(filters)):
        # res1 = process(img, filters[i])
        accum = np.zeros_like(img)
        for kern in filters[i]:
            fimg = cv2.filter2D(img, cv2.CV_8UC1, kern)
            accum = np.maximum(accum, fimg, accum)
        res.append(np.asarray(accum))

    #用于绘制滤波效果
    plt.figure(2)
    for temp in range(len(res)):
        plt.subplot(len(THETA_RADIUS_GROUP), len(GK_SIZES),temp+1)
        plt.imshow(res[temp], cmap='gray')
    plt.show()
    return res  # 返回滤波结果,结果为24幅图，按照gabor角度排列


def _filter_imgs(src_dir, selected_names=None):
    from imgp_agent.imgp_common import FileHelper
    filenames = FileHelper.find_all_images(src_dir)
    filenames = sorted(filenames)

    filters = build_filters()

    for filename in filenames:
        if selected_names is not None:
            if os.path.basename(filename) not in selected_names:
                continue 
    
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        getGabor(image, filters)
        # if image is None:
        #     print("not able to mark hair on", filename)

        # FileHelper.save_output_image(image, src_dir, filename, "hair")

def _get_root_dir():
    import os 

    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    return dir_root

def _add_root_in_sys_path():
    import sys 

    dir_root = _get_root_dir()
    if dir_root not in sys.path:
        sys.path.append(dir_root)

def do_exp():
    selected_names = None 
    #selected_names = ["icl_image5.jpeg"]

    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    src_dir = os.sep.join([_get_root_dir(), "utils_inspect", "_sample_imgs"])

    _filter_imgs(src_dir, selected_names=selected_names)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
