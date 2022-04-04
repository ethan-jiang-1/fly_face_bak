
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

    def _get_edge_b2w(self, img, edge_type):
        if edge_type == "3L2":
            img_edge = cv2.Canny(img, 127, 255, L2gradient=True)
        elif edge_type == "5L2":
            img_edge = cv2.Canny(img, 127, 255, apertureSize=5, L2gradient=True)
        elif edge_type == "7L2":
            img_edge = cv2.Canny(img, 127, 255, apertureSize=5, L2gradient=True)
        return img_edge

    def _get_edge_w2b(self, img, edge_type):
        img_w2b = cv2.bitwise_not(img)

        if edge_type == "3L2":
            img_edge = cv2.Canny(img_w2b, 127, 255, L2gradient=True)
        elif edge_type == "5L2":
            img_edge = cv2.Canny(img_w2b, 127, 255, apertureSize=5, L2gradient=True)
        elif edge_type == "7L2":
            img_edge = cv2.Canny(img_w2b, 127, 255, apertureSize=5, L2gradient=True)
        return img_edge

    def make_aug_images(self, img):
        img_unified = self.resize_to_unified(img)
        img_blur3 = cv2.GaussianBlur(img_unified, (3,3), cv2.BORDER_DEFAULT)
        _, img_blur3 = cv2.threshold(img_blur3, 127, 255, cv2.THRESH_BINARY)
        img_blur5 = cv2.GaussianBlur(img_unified, (5,5), cv2.BORDER_DEFAULT)
        _, img_blur5 = cv2.threshold(img_blur5, 127, 255, cv2.THRESH_BINARY)        
        img_blur7 = cv2.GaussianBlur(img_unified, (7,7), cv2.BORDER_DEFAULT)
        _, img_blur7 = cv2.threshold(img_blur7, 127, 255, cv2.THRESH_BINARY)
        img_blur9 = cv2.GaussianBlur(img_unified, (9,9), cv2.BORDER_DEFAULT)
        _, img_blur9 = cv2.threshold(img_blur9, 127, 255, cv2.THRESH_BINARY)

        imgs_org = [img_unified, img_blur3, img_blur5, img_blur7, img_blur9]
        names_org = ["unified", "blur3", "blur5", "blur7", "blur9"]

        imgs_aug = []
        imgs_map = {}

        for img, name in zip(imgs_org, names_org):
            img_edge = self._get_edge_b2w(img, edge_type="7L2")
            imgs_edged_full = self._shift_edge_full(img, img_edge)

            #imgs_edged_half_right = self._shift_edge_half_right(img, img_edge)
            #imgs_edged_half_left = self._shift_edge_half_left(img, img_edge)
            imgs_map[name] = imgs_edged_full
            imgs_aug.extend(imgs_edged_full)

        if self.debug:
            names = []
            for img in imgs_aug:
                area = cv2.countNonZero(img)
                names.append(str(area))

            self.plot_imgs_grid(imgs_aug, names=names, mod_num=len(imgs_map["unified"]), set_axis_off=True)

            # self.plot_imgs_grid([imgs_edged_full[0], imgs_edged_half_right[0], imgs_edged_half_left[0],
            #                     imgs_edged_full[-1], imgs_edged_half_right[-1], imgs_edged_half_left[-1]], mod_num=3)

        return imgs_aug

    def check_edges_1(self, img):

        img_org = self.resize_to_unified(img)
        img_blur1 = cv2.GaussianBlur(img_org, (5,5), cv2.BORDER_DEFAULT)
        _, img_blur1 = cv2.threshold(img_blur1, 127, 255, cv2.THRESH_BINARY)
        img_blur2 = cv2.GaussianBlur(img_org, (9,9), cv2.BORDER_DEFAULT)
        _, img_blur2 = cv2.threshold(img_blur2, 127, 255, cv2.THRESH_BINARY)

        img_edge0 = self._get_edge_b2w(img_org, edge_type="7L2")
        img_edge1 = self._get_edge_b2w(img_blur1, edge_type="7L2")
        img_edge2 = self._get_edge_b2w(img_blur2, edge_type="7L2")

        self.plot_imgs([img_org, img_edge0, img_blur1, img_edge1, img_blur2, img_edge2])

    def check_edges_2(self, img):
        imgs = []
        names = []
        img_unified = self.resize_to_unified(img)
        imgs.append(img_unified)
        names.append("org")
        
        img_edged1 = cv2.Canny(img_unified, 127, 255)
        imgs.append(img_edged1)
        names.append("3")

        img_edged2 = cv2.Canny(img_unified, 127, 255, apertureSize=5)
        imgs.append(img_edged2)
        names.append("5")

        img_edged3 = cv2.Canny(img_unified, 127, 255, apertureSize=7)
        imgs.append(img_edged3)
        names.append("7")

        img_edged11 = cv2.Canny(img_unified, 127, 255, L2gradient=True)
        imgs.append(img_edged11)
        names.append("3,L3")

        img_edged12 = cv2.Canny(img_unified, 127, 255, apertureSize=5, L2gradient=True)
        imgs.append(img_edged12)
        names.append("5,L2")

        img_edged13 = cv2.Canny(img_unified, 127, 255, apertureSize=7, L2gradient=True)
        imgs.append(img_edged13)
        names.append("7,L2")

        self.plot_imgs(imgs, names=names)

def do_exp(filename):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not filename.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        filename = "{}/{}".format(dir_root, filename)
    
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
    dg = DgEdge(debug=True)
    imgs_aug = dg.make_aug_images(img)
    print(len(imgs_aug))
    #dg.check_edges(img)
    #dg.check_edges_1(img)
    del dg 


if __name__ == '__main__':

    #filename = "_reserved_output_hair_styles/01/01-001.jpg"
    filename = "_reserved_output_hair_styles/05/05-006.jpg"
    do_exp(filename)
