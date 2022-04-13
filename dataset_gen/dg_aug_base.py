import sys
import os
import cv2
import numpy as np
import albumentations as A

dir_parent = os.path.dirname(os.path.dirname(__file__))
if dir_parent not in sys.path:
    sys.path.append(dir_parent)

from utils.file_helper import FileHelper
from utils.plot_helper import PlotHelper


UNIFIED_IMG_SIZE = (112, 112)

class DgAugBase():
    ET_3L2 = "3L2"
    ET_5L2 = "5L2"
    ET_7L2 = "7L2"

    TN_COMBINEA = "combine_a"
    TN_COMBINEB = "combine_b"

    def __init__(self, bin_img=True, debug=False):
        self.debug = debug
        self.bin_img = bin_img

        self.transforms = {}

    def shift_edge_full(self, img):
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
                img_overlap = self.threshold_img(img_overlap)
                imgs.append(img_overlap)
        return imgs

    def shift_edge_half_left(self, img):
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
                img_overlap = self.threshold_img(img_overlap)
                imgs.append(img_overlap)
        return imgs

    def shift_edge_half_right(self, img):
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
                img_overlap = self.threshold_img(img_overlap)
                imgs.append(img_overlap)
        return imgs

    def get_edge_b2w(self, img, edge_type):
        img_edge = None
        if edge_type == self.ET_3L2:
            img_edge = cv2.Canny(img, 127, 255, L2gradient=True)
        elif edge_type == self.ET_5L2:
            img_edge = cv2.Canny(img, 127, 255, apertureSize=5, L2gradient=True)
        elif edge_type == self.ET_7L2:
            img_edge = cv2.Canny(img, 127, 255, apertureSize=5, L2gradient=True)
        else:
            raise ValueError("{} not supported".format(edge_type))
        return img_edge

    def get_edge_w2b(self, img, edge_type):
        img_w2b = cv2.bitwise_not(img)

        if edge_type == self.ET_3L2:
            img_edge = cv2.Canny(img_w2b, 127, 255, L2gradient=True)
        elif edge_type == self.ET_5L2:
            img_edge = cv2.Canny(img_w2b, 127, 255, apertureSize=5, L2gradient=True)
        elif edge_type == self.ET_7L2:
            img_edge = cv2.Canny(img_w2b, 127, 255, apertureSize=5, L2gradient=True)
        else:
            raise ValueError("{} not supported".format(edge_type))
        return img_edge

    def get_transforms(self, transform_name):
        if len(self.transforms) == 0:
            transform = A.Compose([A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), rotate_limit=(-5,5), scale_limit=(0.0, 0.0), p=1.0)])
            self.transforms[self.TN_COMBINEA] = transform

            transform = A.Compose([A.GridDistortion(num_steps=1, distort_limit=0.1, p=1.0)])
            self.transforms[self.TN_COMBINEB] = transform
        return self.transforms[transform_name]

    def transform_img(self, img, transform_name):
        transform = self.get_transforms(transform_name)
        result = transform(image=img)
        img_transformed = result["image"]
        img_transformed = self.threshold_img(img_transformed)
        img_resized = cv2.resize(img_transformed, UNIFIED_IMG_SIZE)
        return img_resized

    def get_empty_unified(self):
        img = np.zeros(UNIFIED_IMG_SIZE, dtype=np.uint8)
        return img

    def resize_to_unified(self, img):
        img_th = self.threshold_img(img)
        img_resized = cv2.resize(img_th, UNIFIED_IMG_SIZE)
        return img_resized

    def plot_imgs(self, imgs, names=None, title=None):
        return PlotHelper.plot_imgs(imgs, names=names, title=title)

    def plot_imgs_grid(self, imgs, names=None, title=None, mod_num=4, figsize=(10, 8), set_axis_off=False):
        return PlotHelper.plot_imgs_grid(imgs, names=names, title=title, mod_num=mod_num, figsize=figsize, set_axis_off=set_axis_off)

    def plot_aug_imags_map(self, imgs_map):
        _, min_len = self.check_imgs_map_size(imgs_map)

        names = []
        imgs_aug = []
        for _, imgs in imgs_map.items():
            for idx, img in enumerate(imgs):
                if idx <= min_len-1:
                    area = cv2.countNonZero(img)
                    names.append(str(area))
                    imgs_aug.append(img)

        self.plot_imgs_grid(imgs_aug, names=names, mod_num=min_len, set_axis_off=True)

    def check_imgs_map_size(self, imgs_map):
        aug_len = []
        for _, imgs in imgs_map.items():
            aug_len.append(len(imgs))
    
        if len(aug_len) == 0:
            return 0, 0
        np_aug_len = np.array(aug_len)
        min_aug_len = np_aug_len.min()
        total_aug_len = np_aug_len.sum()
        return total_aug_len, min_aug_len        

    def make_aug_edge_shift(self, img_unified, aug_types=["shift_full", "shift_right", "shift_left"], num_blur=4):
        imgs_org = [img_unified]
        names_org = ["unified"]

        for i in range(num_blur):
            blur_core = 3 + i * 2
            
            img_blur = cv2.GaussianBlur(img_unified, (blur_core, blur_core), cv2.BORDER_DEFAULT)
            img_blur = self.threshold_img(img_blur)

            imgs_org.append(img_blur)
            names_org.append("blur{}".format(blur_core))

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

    def make_aug_transform(self, aug_imgs_map, aug_types=[]):
        trs_imgs_map = {}
        for key, imgs in aug_imgs_map.items():
            if "transform_a" in aug_types:
                key_trs = "{}_trsa".format(key)
                trs_imgs_map[key_trs] =[]
                for img in imgs:
                    img_transformed = self.transform_img(img, self.TN_COMBINEA)
                    trs_imgs_map[key_trs].append(img_transformed)
            if "transform_b" in aug_types:
                key_trs = "{}_trsb".format(key)
                trs_imgs_map[key_trs] =[]
                for img in imgs:
                    img_transformed = self.transform_img(img, self.TN_COMBINEB)
                    trs_imgs_map[key_trs].append(img_transformed)
        return trs_imgs_map

    def threshold_img(self, img):
        if img.max() == img.min():
            return np.zeros_like(img)
        
        if self.bin_img:
            _, img_th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        else:
            img_float  = img.astype('float')
            img_float -= img_float.min()
            img_float /= img_float.max()
            img_float *= 255
            img_th = img_float.astype('uint8')
        return img_th

if __name__ == '__main__':
    print(FileHelper, PlotHelper)
