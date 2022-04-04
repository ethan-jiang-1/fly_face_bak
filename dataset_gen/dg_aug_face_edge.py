
# REF https://blog.csdn.net/qq_45769063/article/details/107339132
import sys
import os 
import cv2
import numpy as np

try:
    from dg_aug_base import DgAugBase
except:
    from .dg_aug_base import DgAugBase


class DgAugFaceEdge(DgAugBase):
    def __init__(self, debug=True):
        super(DgAugFaceEdge, self).__init__()
        self.debug = debug

    def _make_aug_edge_shift(self, img_unified, aug_types=["shift_full", "shift_right", "shift_left"]):
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

        imgs_aug_map = {}

        for img, name in zip(imgs_org, names_org):
            if "shift_full" in aug_types:
                imgs_edged_full = self.shift_edge_full(img)
                imgs_aug_map[name + "_shift_full"] = imgs_edged_full

            if "shift_right" in aug_types:
                imgs_edged_half_right = self.shift_edge_half_right(img)
                imgs_aug_map[name + "_shift_right"] = imgs_edged_half_right

            if "shift_left" in aug_types:
                imgs_edged_half_left = self.shift_edge_half_left(img)
                imgs_aug_map[name + "_shift_left"] = imgs_edged_half_left

        return imgs_aug_map

    def _plot_aug_imags(self, imgs_map, min_len):
        names = []
        imgs_aug = []
        for _, imgs in imgs_map.items():
            for idx, img in enumerate(imgs):
                if idx <= min_len-1:
                    area = cv2.countNonZero(img)
                    names.append(str(area))
                    imgs_aug.append(img)

        self.plot_imgs_grid(imgs_aug, names=names, mod_num=min_len, set_axis_off=True)

    def make_aug_images(self, img, aug_types=["shift_full", "shift_right", "shift_left", "transform"]):
        img_unified = self.resize_to_unified(img)

        #aug_imgs_map = self._make_aug_edge_shift(img_unified, aug_types=aug_types)
        #aug_imgs_map = self._make_aug_edge_shift(img_unified, aug_types=["shift_right"])
        aug_imgs_map = self._make_aug_edge_shift(img_unified, aug_types=["shift_left"])

        aug_len = []
        trs_len = []
        trs_imgs_map = {}
        for key, imgs in aug_imgs_map.items():
            key_trs = "{}_trs".format(key)
            trs_imgs_map[key_trs] =[]
            aug_len.append(len(imgs))
            print("{}: {}".format(key, len(imgs)))
            if "transform" in aug_types:
                for img in imgs:
                    img_transformed = self.transform_img(img, self.TN_COMBINEA)
                    trs_imgs_map[key_trs].append(img_transformed)
            trs_len.append(len(trs_imgs_map[key_trs]))
    
        np_aug_len = np.array(aug_len)
        min_aug_len = np_aug_len.min()
        total_aug_len = np_aug_len.sum()
        print("aug_imgs_map total_len: {}/{} for {}".format(total_aug_len, min_aug_len, aug_types))
    
        np_trs_len = np.array(trs_len)
        min_trs_len = np_trs_len.min()
        total_trs_len = np_trs_len.sum()
        print("trs_imgs_map total_len: {}/{} for {}".format(total_trs_len, min_trs_len, aug_types))

        if self.debug:
            self._plot_aug_imags(aug_imgs_map, min_aug_len)
            self._plot_aug_imags(trs_imgs_map, min_trs_len)

        return aug_imgs_map, trs_imgs_map

    def check_edges(self, img):

        img_org = self.resize_to_unified(img)
        img_blur1 = cv2.GaussianBlur(img_org, (5,5), cv2.BORDER_DEFAULT)
        _, img_blur1 = cv2.threshold(img_blur1, 127, 255, cv2.THRESH_BINARY)
        img_blur2 = cv2.GaussianBlur(img_org, (9,9), cv2.BORDER_DEFAULT)
        _, img_blur2 = cv2.threshold(img_blur2, 127, 255, cv2.THRESH_BINARY)

        img_edge0 = self.get_edge_b2w(img_org, edge_type=self.ET_3L2)
        img_edge1 = self.get_edge_b2w(img_blur1, edge_type=self.ET_5L2)
        img_edge2 = self.get_edge_b2w(img_blur2, edge_type=self.ET_7L2)

        self.plot_imgs([img_org, img_edge0, img_blur1, img_edge1, img_blur2, img_edge2])

        imgs = [img_org]
        for i in range(5):
            img_transformed = self.transform_img(img_org, self.TN_COMBINEA)
            imgs.append(img_transformed)
        self.plot_imgs(imgs)


def do_exp(filename):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not filename.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        filename = "{}/{}".format(dir_root, filename)
    
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
    dg = DgAugFaceEdge(debug=True)

    aug_imgs_map, trs_imgs_map = dg.make_aug_images(img)
    #dg.check_edges(img)

    del dg 


if __name__ == '__main__':

    #filename = "_reserved_output_hair_styles/01/01-001.jpg"
    filename = "_reserved_output_hair_styles/05/05-006.jpg"
    do_exp(filename)
