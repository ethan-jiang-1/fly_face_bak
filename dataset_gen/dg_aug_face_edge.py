import sys
import os 
import cv2

try:
    from dg_aug_base import DgAugBase
except:
    from .dg_aug_base import DgAugBase


class DgAugFaceEdge(DgAugBase):
    def __init__(self, debug=False):
        super(DgAugFaceEdge, self).__init__(debug=debug)

    def make_aug_images(self, img, aug_types=("shift_full", "shift_right", "shift_left", "transform_a", "transform_b")):
        img_unified = self.resize_to_unified(img)

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

def do_exp(filename):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not filename.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        filename = "{}/{}".format(dir_root, filename)
    
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)

    dg = DgAugFaceEdge(debug=True)
    aug_imgs_map, trs_imgs_map = dg.make_aug_images(img, aug_types=("shift_left", "transform_a"))
    print(len(aug_imgs_map), len(trs_imgs_map))
 
    del dg 


if __name__ == '__main__':

    #filename = "_reserved_output_hair_styles/01/01-001.jpg"
    filename = "_reserved_output_hair_styles/05/05-006.jpg"
    do_exp(filename)
