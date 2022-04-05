#!/usr/bin/env python

import os
import PySimpleGUI as sg
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

    PlotHelper.plot_imgs_grid(imgs, names=names, mod_num=num_in_group, figsize=(10, 3), set_axis_off=True)


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


def analysis_all_files():
    #dir_root = os.path.dirname(__file__)
    src_dirs = [] 
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/01")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/02")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/03")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/04")
    src_dirs.append("_dataset_org_hair_styles/Version 1.1/05")

    dst_dir = "_reserved_output_hair_styles"

    do_extract_hairstyles(src_dirs, dst_dir)


def analysis_selected_folder():
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('选择一个目录', size=(15, 1), auto_size_text=True, justification='left')], 
              [sg.InputText(size=(80, 1)), sg.FolderBrowse(initial_folder=os.path.dirname(__file__))], 
              [sg.Button('OK'), sg.Button('Exit')]]
    
    window = sg.Window('发型分析', layout)
    
    while True:
        event, values = window.read()
        print(f'Event: {event}, Values: {values}')
        
        if event == 'OK':
            if (values[0] == ''):
                sg.popup('请选择发型图片所在的目录！')
            elif not os.path.exists(values[0]):
                sg.popup('目录不存在，请重新选择！')
            else:
                src_dirs = [values[0]]
                dst_dir = "_reserved_output_hair_styles"
                do_extract_hairstyles(src_dirs, dst_dir)
        elif event in (None, 'Exit'):
            break
    window.close()

 
if __name__ == '__main__':
    # analysis_all_files()
    analysis_selected_folder()
