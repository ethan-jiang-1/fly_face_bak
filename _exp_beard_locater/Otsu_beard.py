# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 00:53:24 2022

@author: Administrator
"""
import os 
import cv2
from matplotlib import pyplot as plt

def plot_beard(img_color):
    #img = cv2.imread(filename, 0)
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    print("img.shape", img.shape)

    # 固定阈值法
    ret1, th1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Otsu阈值法
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 先进行高斯滤波，再使用Otsu阈值法
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [img, 0, th1, img, 0, th2, blur, 0, th3]
    titles = ['Original', 'Histogram', 'Global(v=100) th1',
              'Original', 'Histogram', "Otsu's th2",
              'Gaussian filtered Image', 'Histogram', "Otsu's th3"]

    for i in range(3):
        # 绘制原图
        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3], fontsize=8)
        plt.xticks([]), plt.yticks([])
        
        # 绘制直方图plt.hist, ravel函数将数组降成一维
        plt.subplot(3, 3, i * 3 + 2)
        plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1], fontsize=8)
        plt.xticks([]), plt.yticks([])
        
        # 绘制阈值图
        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2], fontsize=8)
        plt.xticks([]), plt.yticks([])
    plt.show()


def _mark_beard_imgs(src_dir, selected_names=None):
    from imgp_agent.imgp_common import FileHelper
    filenames = FileHelper.find_all_images(src_dir)


    for filename in filenames:
        if selected_names is not None:
            if os.path.basename(filename) not in selected_names:
                continue 
    
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        plot_beard(image)
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

    _mark_beard_imgs(src_dir, selected_names=selected_names)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()