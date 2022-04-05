import sys
import os 
import random
import cv2

try:
    from dg_aug_base import DgAugBase
except:
    from .dg_aug_base import DgAugBase


class DgAugEmpty(DgAugBase):
    def __init__(self, debug=False):
        super(DgAugEmpty, self).__init__(debug=debug)

    def gen_empty_img(self, noise_theshold=240):
        img_empty = self.get_empty_unified()
        for x in range(img_empty.shape[1]):
            for y in range(img_empty.shape[0]):
                img_empty[x, y] = (random.randint(0, 255))
        _, img_empty_b = cv2.threshold(img_empty, noise_theshold, 255, cv2.THRESH_BINARY)        
        return img_empty_b

    def make_aug_images(self, noise_theshold=240, aug_types=("shift_full", "shift_right", "shift_left", "transform_a", "transform_b")):
        img_unified = self.gen_empty_img(noise_theshold=noise_theshold)

        aug_imgs_map = self.make_aug_edge_shift(img_unified, aug_types=aug_types)
        trs_imgs_map = self.make_aug_transform(aug_imgs_map, aug_types=aug_types)

        total_aug_len, min_aug_len = self.check_imgs_map_size(aug_imgs_map)
        if self.debug:
            print("aug_imgs_map total_len: {}/{} for {}".format(total_aug_len, min_aug_len, aug_types))
    
        total_trs_len, min_trs_len = self.check_imgs_map_size(trs_imgs_map)
        if self.debug:
            print("trs_imgs_map total_len: {}/{} for {}".format(total_trs_len, min_trs_len, aug_types))

        if self.debug:
            self.plot_aug_imags_map(aug_imgs_map)
            self.plot_aug_imags_map(trs_imgs_map)

        return aug_imgs_map, trs_imgs_map

def do_exp():
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)
    
    dg = DgAugEmpty(debug=True)
    aug_imgs_map, trs_imgs_map = dg.make_aug_images(noise_theshold=255-8, aug_types=("shift_full", "transform_a"))
    print(len(aug_imgs_map), len(trs_imgs_map))
 
    del dg 


if __name__ == '__main__':
    do_exp()
