
# REF https://blog.csdn.net/qq_45769063/article/details/107339132
import sys
import os 
import cv2
import numpy as np

try:
    from dg_base import DgBase
except:
    from .dg_base import DgBase


class DgEdge(DgBase):
    def __init__(self, debug=True):
        super(DgEdge, self).__init__()
        self.debug = debug

    def _shift_edge_full(self, img, img_edged):
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

    def _shift_edge_half_left(self, img, img_edged):
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

    def _shift_edge_half_right(self, img, img_edged):
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

    def aug(self, img):

        img_unified = self.resize_to_unified(img)
        img_edged = cv2.Canny(img_unified, 127, 255)

        imgs_edged_full = self._shift_edge_full(img_unified, img_edged)
        imgs_edged_half_right = self._shift_edge_half_right(img_unified, img_edged)
        imgs_edged_half_left = self._shift_edge_half_left(img_unified, img_edged)

        imgs = [] # [img, img_unified, img_edged]
        imgs.extend(imgs_edged_full)
        imgs.extend(imgs_edged_half_right)
        imgs.extend(imgs_edged_half_left)

        if self.debug:
            self.plot_imgs_grid(imgs, mod_num=len(imgs_edged_full), set_axis_off=True)

            self.plot_imgs_grid([imgs_edged_full[0], imgs_edged_half_right[0], imgs_edged_half_left[0],
                                imgs_edged_full[-1], imgs_edged_half_right[-1], imgs_edged_half_left[-1]], mod_num=3)

        return imgs


def do_exp(filename):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not filename.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        filename = "{}/{}".format(dir_root, filename)
    
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
    dg = DgEdge(debug=True)
    dg.aug(img)
    del dg 


if __name__ == '__main__':

    filename = "_reserved_output_hair_styles/01/01-001.jpg"
    do_exp(filename)
