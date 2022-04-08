import sys
import os 
import cv2

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

try:
    from dg_aug_base import FileHelper
    from dg_aug_face_edge import DgAugFaceEdge
    from dg_aug_empty import DgAugEmpty
except:
    from .dg_aug_base import FileHelper
    from .dg_aug_face_edge import DgAugFaceEdge
    from .dg_aug_empty import DgAugEmpty

from imgp_agent.face_feature_generator import FaceFeatureGenerator
from utils.colorstr import log_colorstr


class DatasetHairstyleGen():
    def __init__(self, dir_org, dir_dst):

        self.dir_org = dir_org
        self.dir_dst = dir_dst
        self.ffg = FaceFeatureGenerator()

    def __del__(self):
        del self.ffg

    def _prepare_dirs(self):
        if not os.path.isdir(self.dir_org):
            raise ValueError("dataset org dir {} not exist".format(self.dir_org))

        os.makedirs(self.dir_dst, exist_ok=True)

    def _get_img_hair(self, img_pathname):
        imgs, names, _, _ = self.ffg.process_image(img_pathname)
        img_hair = self.ffg.find_img(imgs, names, "hair")
        return img_hair

    def _process_aug_imgs_in_subdir(self, subname):
        dst_dir = "{}/{}".format(self.dir_dst, subname)
        os.makedirs(dst_dir, exist_ok=True)

        dg = DgAugFaceEdge()
        names = os.listdir(self.dir_org + "/" + subname)
        names = sorted(names)
        log_colorstr("yellow", "generating {}...".format(dst_dir))
        cnt = 0

        order_num = -1
        for name in names:
            img_pathname = "{}/{}/{}".format(self.dir_org, subname, name)
            if not FileHelper.is_supported_img_filename(img_pathname):
                continue

            order_num += 1
            img = self._get_img_hair(img_pathname)
            aug_imgs_map, trs_imgs_map = dg.make_aug_images(img)
            cnt += self._save_aug_imgs(aug_imgs_map, trs_imgs_map, dst_dir, subname, order_num)

        log_colorstr("yellow", "generated {} images in {}\n".format(cnt, dst_dir))

    def _save_aug_imgs(self, aug_imgs_map, trs_imgs_map, dst_dir, subname, order_num):
        cnt =0
        gp_num = -1
        for key, imgs in aug_imgs_map.items():
            gp_num += 1
            for idx, img in enumerate(imgs):
                name = "{}/{}_{:03}_{:04}_{}.jpg".format(dst_dir, subname, order_num*100 + gp_num, idx, key)
                cv2.imwrite(name, img)
                cnt += 1 

        gp_num = -1
        for key, imgs in trs_imgs_map.items():
            gp_num += 1
            for idx, img in enumerate(imgs):
                name = "{}/{}_{:03}_{:04}_{}.jpg".format(dst_dir, subname, order_num*100 + gp_num, idx, key)
                cv2.imwrite(name, img)
                cnt += 1
        return cnt

    def _process_aug_empty_imgs_in_subdir(self, subname):
        dst_dir = "{}/{}".format(self.dir_dst, subname)
        os.makedirs(dst_dir, exist_ok=True)
        log_colorstr("yellow", "generating {}...".format(dst_dir))
        cnt = 0

        dg = DgAugEmpty()
        for order_num in range(18):
            aug_imgs_map, trs_imgs_map = dg.make_aug_images(noise_theshold=255-8-order_num)
            cnt += self._save_aug_imgs(aug_imgs_map, trs_imgs_map, dst_dir, subname, order_num)

        log_colorstr("yellow", "generated {} images in {}".format(cnt, dst_dir))

    def gen(self):
        self._prepare_dirs()
        subnames = os.listdir(self.dir_org)
        subnames = sorted(subnames)
        for subname in subnames:
            dir_sub = "{}/{}".format(self.dir_org, subname)
            if not os.path.isdir(dir_sub):
                continue
            self._process_aug_imgs_in_subdir(subname)
        self._process_aug_empty_imgs_in_subdir("00")


def do_exp(dir_org, dir_dst):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not dir_org.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        dir_org = "{}/{}".format(dir_root, dir_org)

    if not dir_dst.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        dir_dst = "{}/{}".format(dir_root, dir_dst)

    dhg = DatasetHairstyleGen(dir_org, dir_dst)
    dhg.gen()
    del dhg


if __name__ == '__main__':

    dir_org = "dataset_org_hair_styles/Version 1.2"
    dir_dst = "_dataset_hair_styles"
    do_exp(dir_org, dir_dst)
