
# REF https://blog.csdn.net/qq_45769063/article/details/107339132
import sys
import os 
import cv2
import numpy as np

try:
    from dg_aug_base import DgAugBase
except:
    from .dg_aug_base import DgAugBase


class DgAugBeardEdge(DgAugBase):
    def __init__(self, debug=True):
        super(DgAugBeardEdge, self).__init__()
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

    def make_aug_images(self, img, aug_types=["shift_full", "shift_right", "shift_left"]):
        img_unified = self.resize_to_unified(img)

        #imgs_map = self._make_aug_edge_shift(img_unified, aug_types=aug_types)
        imgs_map = self._make_aug_edge_shift(img_unified, aug_types=["shift_right"])
        #imgs_map = self._make_aug_edge_shift(img_unified, aug_types=["shift_left"])

        imgs_aug = []
        imgs_len = []
        for _, imgs in imgs_map.items():
            imgs_aug.extend(imgs)
            imgs_len.append(len(imgs))
        np_imgs_len = np.array(imgs_len)
        min_len = np_imgs_len.min()
        total_len = np_imgs_len.sum()
        print("total_len: {} min_len {} for {}".format(total_len, min_len, aug_types))
    
        if self.debug:
            self._plot_aug_imags(imgs_map, min_len)

        return imgs_aug


def do_exp(filename):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not filename.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        filename = "{}/{}".format(dir_root, filename)
    
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
    dg = DgAugBeardEdge(debug=True)
    imgs_aug = dg.make_aug_images(img)
    print(len(imgs_aug))

    #dg.check_edges(img)

    del dg 


if __name__ == '__main__':

    #filename = "_reserved_output_feature_gen/brd_image5_beard.jpg"
    filename = "_reserved_output_feature_gen/sun_me_beard.jpg"
    do_exp(filename)
