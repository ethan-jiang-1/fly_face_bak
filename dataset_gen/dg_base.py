
import sys
import os
import cv2

dir_parent = os.path.dirname(os.path.dirname(__file__))
if dir_parent not in sys.path:
    sys.path.append(dir_parent)

from utils.file_helper import FileHelper
from utils.plot_helper import PlotHelper


IMG_SIZE = (112, 112)
#IMG_SIZE = (64, 32)

class DgBase():
    def __init__(self):
        pass

    def resize_to_unified(self, img):
        _, img_b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_resized = cv2.resize(img_b, IMG_SIZE)
        return img_resized

    def plot_imgs(self, imgs, names=None, title=None):
        return PlotHelper.plot_imgs(imgs, names=names, title=title)

    def plot_imgs_grid(self, imgs, names=None, title=None, mod_num=4, figsize=(10, 8), set_axis_off=False):
        return PlotHelper.plot_imgs_grid(imgs, names=names, title=title, mod_num=mod_num, figsize=figsize, set_axis_off=set_axis_off)

if __name__ == '__main__':
    print(FileHelper, PlotHelper)
