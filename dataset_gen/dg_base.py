
from abc import ABC, abstractmethod
import sys
import os
import cv2
import numpy as np

dir_parent = os.path.dirname(os.path.dirname(__file__))
if dir_parent not in sys.path:
    sys.path.append(dir_parent)

from utils.file_helper import FileHelper
from utils.plot_helper import PlotHelper


UNIFIED_IMG_SIZE = (112, 112)

class EdgeShifterMixIn(object):
    ET_3L2 = "3L2"
    ET_5L2 = "5L2"
    ET_7L2 = "7L2"

    @classmethod
    def shift_edge_full(cls, img):
        imgs = []

        height = img.shape[0]
        width = img.shape[1]
        img_core = np.zeros((height - 2, width -2), dtype=img.dtype)
        img_core[:, :] = img[1:height-1, 1:width-1]

        for hc in [0, 1, 2]:
            for wc in [0, 1, 2]:
                if hc == 1 and wc == 1:
                    continue 
                img_overlap = img.copy()
                img_overlap[hc:height-2+hc, wc:width-2+wc] += img_core
                _, img_overlap = cv2.threshold(img_overlap, 127, 255, cv2.THRESH_BINARY)
                imgs.append(img_overlap)
        return imgs

    @classmethod
    def shift_edge_half_left(cls, img):
        imgs = []

        height = img.shape[0]
        width = int(img.shape[1] / 2)
        img_core = np.zeros((height - 2, width - 2), dtype=img.dtype)
        img_core[:, :] = img[1:height-1, 1:width-1]

        for hc in [0, 1, 2]:
            for wc in [0, 1, 2]:
                if hc == 1 and wc == 1:
                    continue 
                img_overlap = img.copy()
                img_overlap[hc:height-2+hc, wc:width-2+wc] += img_core
                _, img_overlap = cv2.threshold(img_overlap, 127, 255, cv2.THRESH_BINARY)
                imgs.append(img_overlap)
        return imgs

    @classmethod
    def shift_edge_half_right(cls, img):
        imgs = []

        height = img.shape[0]
        width = int(img.shape[1] / 2)
        img_core = np.zeros((height - 2, width - 2), dtype=img.dtype)
        img_core[:, :] = img[1:height-1, width+1:width+width-1]

        for hc in [0, 1, 2]:
            for wc in [0, 1, 2]:
                if hc == 1 and wc == 1:
                    continue 
                img_overlap = img.copy()
                img_overlap[hc:height-2+hc, width+wc:width+width-2+wc] += img_core
                _, img_overlap = cv2.threshold(img_overlap, 127, 255, cv2.THRESH_BINARY)
                imgs.append(img_overlap)
        return imgs

    @classmethod
    def get_edge_b2w(cls, img, edge_type):
        img_edge = None
        if edge_type == cls.ET_3L2:
            img_edge = cv2.Canny(img, 127, 255, L2gradient=True)
        elif edge_type == cls.ET_5L2:
            img_edge = cv2.Canny(img, 127, 255, apertureSize=5, L2gradient=True)
        elif edge_type == cls.ET_7L2:
            img_edge = cv2.Canny(img, 127, 255, apertureSize=5, L2gradient=True)
        else:
            raise ValueError("{} not supported".format(edge_type))
        return img_edge

    @classmethod
    def get_edge_w2b(cls, img, edge_type):
        img_w2b = cv2.bitwise_not(img)

        if edge_type == cls.ET_3L2:
            img_edge = cv2.Canny(img_w2b, 127, 255, L2gradient=True)
        elif edge_type == cls.ET_5L2:
            img_edge = cv2.Canny(img_w2b, 127, 255, apertureSize=5, L2gradient=True)
        elif edge_type == cls.ET_7L2:
            img_edge = cv2.Canny(img_w2b, 127, 255, apertureSize=5, L2gradient=True)
        else:
            raise ValueError("{} not supported".format(edge_type))
        return img_edge


class DgBase(ABC, EdgeShifterMixIn):
    def __init__(self):
        pass

    @abstractmethod
    def make_aug_images(self, img):
        return []

    def resize_to_unified(self, img):
        _, img_b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img_resized = cv2.resize(img_b, UNIFIED_IMG_SIZE)
        return img_resized

    def plot_imgs(self, imgs, names=None, title=None):
        return PlotHelper.plot_imgs(imgs, names=names, title=title)

    def plot_imgs_grid(self, imgs, names=None, title=None, mod_num=4, figsize=(10, 8), set_axis_off=False):
        return PlotHelper.plot_imgs_grid(imgs, names=names, title=title, mod_num=mod_num, figsize=figsize, set_axis_off=set_axis_off)


if __name__ == '__main__':
    print(FileHelper, PlotHelper)
