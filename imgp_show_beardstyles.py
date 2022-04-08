#!/usr/bin/env python

import os
import cv2
from imgp_agent.face_feature_generator import FaceFeatureGenerator
from imgp_agent.imgp_common import FileHelper, PlotHelper


def _plot_all_imgs_by_range(all_hair_imgs, idx_range=(0, 10), col_size=10):
    imgs = []
    names = []
    for key, tis in all_hair_imgs.items():
        for ndx, ti in enumerate(tis):
            if ndx >= idx_range[0] and ndx < idx_range[1]:
                title, img = ti
                name = "{}.{}".format(key, title.split(".")[0])
                names.append(name)
                imgs.append(img)

    PlotHelper.plot_imgs_grid(imgs, names=names, mod_num=col_size, figsize=(14, 9), set_axis_off=True)


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


def do_extract_beardstyles(src_dirs, dst_dir, plot_img=True, save_img=True):
    from utils.colorstr import log_colorstr
    ffg = FaceFeatureGenerator()

    all_beard_imgs = {}
    for src_dir in src_dirs:
        beard_imgs = []
        all_beard_imgs[os.path.basename(src_dir)] = beard_imgs

        filenames = FileHelper.find_all_images(src_dir)
        if len(filenames) == 0:
            log_colorstr("red", "no files find in {}".format(src_dir))
            
        #print(filenames)
        filenames = sorted(filenames)
        for filename in filenames:
            img = ffg.process_image_for(filename, "beard")
            title = os.path.basename(filename)
            beard_imgs.append((title, img))

    if save_img:                    
        _save_all_imgs(all_beard_imgs, dst_dir)
    if plot_img:
        _plot_all_imgs_by_range(all_beard_imgs, idx_range=(0, 10), col_size=10)
        _plot_all_imgs_by_range(all_beard_imgs, idx_range=(10, 20), col_size=10)

    del ffg


def analysis_all_files():
    #dir_root = os.path.dirname(__file__)
    src_dirs = [] 
    src_dirs.append("dataset_org_beard_styles/Beard Version 1.1/01")
    src_dirs.append("dataset_org_beard_styles/Beard Version 1.1/02")
    src_dirs.append("dataset_org_beard_styles/Beard Version 1.1/03")

    dst_dir = "_reserved_output_beard_styles"

    do_extract_beardstyles(src_dirs, dst_dir)

 
if __name__ == '__main__':
    analysis_all_files()

