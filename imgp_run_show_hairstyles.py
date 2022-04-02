import os
#import numpy as np

from imgp_agent.face_feature_generator import FaceFeatureGenerator
from imgp_agent.imgp_common import FileHelper, PlotHelper

def _plot_all_imgs(all_hair_imgs):
    # cat_len = len(all_hair_imgs.keys())
    # ti_len = []
    # for key, tis in all_hair_imgs.items():
    #     ti_len.append(len(tis))
    # #min = np.array(ti_len).min()
    min = 10

    imgs = []
    names = []
    for key, tis in all_hair_imgs.items():
        for ndx, ti in enumerate(tis):
            if ndx > min - 1:
                break
            title, img = ti
            name = "{}.{}".format(key, title.split(".")[0])
            names.append(name)
            imgs.append(img)

    PlotHelper.plot_imgs_grid(imgs, names=names, mod_num=min, figsize=(10, 8), set_axis_off=True)

def _do_exp_on_dir(src_dirs):
    ffg = FaceFeatureGenerator()

    all_hair_imgs = {}
    for src_dir in src_dirs:
        hair_imgs = []
        all_hair_imgs[os.path.basename(src_dir)] = hair_imgs

        filenames = FileHelper.find_all_images(src_dir)
        #print(filenames)
        filenames = sorted(filenames)
        for filename in filenames:
            imgs, names, title, dt = ffg.process_image(filename)
            for idx, name in enumerate(names):
                if name == "hair":
                    hair_imgs.append((title, imgs[idx]))
                    
    _plot_all_imgs(all_hair_imgs)
    del ffg


def do_exp():
    #dir_root = os.path.dirname(__file__)
    src_dirs = [] 
    src_dirs.append("_reserved_hair_styles/Version 1.0/01")
    src_dirs.append("_reserved_hair_styles/Version 1.0/02")
    src_dirs.append("_reserved_hair_styles/Version 1.0/03")
    src_dirs.append("_reserved_hair_styles/Version 1.0/04")
    src_dirs.append("_reserved_hair_styles/Version 1.0/05")

    _do_exp_on_dir(src_dirs)

if __name__ == '__main__':
    do_exp()
