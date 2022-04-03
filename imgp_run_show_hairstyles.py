import os
#import numpy as np

import cv2
from imgp_agent.face_feature_generator import FaceFeatureGenerator
from imgp_agent.imgp_common import FileHelper, PlotHelper


def _plot_all_imgs(all_hair_imgs, num_in_group=10):
    from utils.colorstr import log_colorstr

    for key, tis in all_hair_imgs.items():
        if len(tis) % num_in_group != 0:
            log_colorstr("red", "num of imgs in {} is {} (can not be divied by {}".format(key, len(tis), num_in_group))

    imgs = []
    names = []
    for key, tis in all_hair_imgs.items():
        for ndx, ti in enumerate(tis):
            if ndx > num_in_group - 1:
                break
            title, img = ti
            name = "{}.{}".format(key, title.split(".")[0])
            names.append(name)
            imgs.append(img)

    PlotHelper.plot_imgs_grid(imgs, names=names, mod_num=num_in_group, figsize=(10, 8), set_axis_off=True)

def _save_all_imgs(all_hair_imgs, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for key, tis in all_hair_imgs.items():
        dst_sub_dir = "{}/{}".format(dst_dir, key)
        os.makedirs(dst_sub_dir, exist_ok=True)
        for ndx, ti in enumerate(tis):
            title, img = ti
            filename = "{}/{}-{}.jpg".format(dst_sub_dir, key, title.split(".")[0])
            img_112 = cv2.resize(img, (112, 112))
            cv2.imwrite(filename, img_112)
            print("{} saved".format(filename))


def do_extract_hairstyles(src_dirs, dst_dir, plot_img=True, save_img=True):
    from utils.colorstr import log_colorstr
    ffg = FaceFeatureGenerator()

    all_hair_imgs = {}
    for src_dir in src_dirs:
        hair_imgs = []
        all_hair_imgs[os.path.basename(src_dir)] = hair_imgs

        filenames = FileHelper.find_all_images(src_dir)
        if len(filenames) == 0:
            log_colorstr("red", "no files find in {}".format(src_dir))
            
        #print(filenames)
        filenames = sorted(filenames)
        for filename in filenames:
            imgs, names, title, dt = ffg.process_image(filename)
            for idx, name in enumerate(names):
                if name == "hair":
                    hair_imgs.append((title, imgs[idx]))

    if save_img:                    
        _save_all_imgs(all_hair_imgs, dst_dir)
    if plot_img:
        _plot_all_imgs(all_hair_imgs)

    del ffg


def do_exp():
    #dir_root = os.path.dirname(__file__)
    src_dirs = [] 
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/01")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/02")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/03")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/04")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/05")

    dst_dir = "_reserved_output_hair_styles"

    do_extract_hairstyles(src_dirs, dst_dir)

if __name__ == '__main__':
    do_exp()
