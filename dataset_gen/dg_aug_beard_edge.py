import sys
import os 
import cv2

try:
    from dg_aug_base import DgAugBase
except:
    from .dg_aug_base import DgAugBase


class DgAugBeardEdge(DgAugBase):
    def __init__(self, debug=False):
        super(DgAugBeardEdge, self).__init__(debug=debug)

    def make_aug_images(self, img, aug_types=["shift_full", "shift_right", "shift_left", "transform_a"]):
        img_unified = self.resize_to_unified(img)

        aug_types=["shift_left", "transform_a"]

        aug_imgs_map = self.make_aug_edge_shift(img_unified, aug_types=aug_types)
        trs_imgs_map = self.make_aug_transform(aug_imgs_map, aug_types=aug_types)

        total_aug_len, min_aug_len = self.check_imgs_map_size(aug_imgs_map)
        print("aug_imgs_map total_len: {}/{} for {}".format(total_aug_len, min_aug_len, aug_types))
    
        total_trs_len, min_trs_len = self.check_imgs_map_size(trs_imgs_map)
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
    dg = DgAugBeardEdge(debug=True)
    imgs_aug = dg.make_aug_images(img)
    print(len(imgs_aug))

    #dg.check_edges(img)

    del dg 


if __name__ == '__main__':

    #filename = "_reserved_output_feature_gen/brd_image5_beard.jpg"
    filename = "_reserved_output_feature_gen/sun_me_beard.jpg"
    do_exp(filename)
