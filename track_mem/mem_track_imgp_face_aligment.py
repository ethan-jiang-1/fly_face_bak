import os 
import sys
import gc
#import numpy as np

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from track_mem.mem_tracker import mem_dump, plot_mem_history
from utils_inspect.sample_images import SampleImages
from imgp_agent.imgp_face_aligment import ImgpFaceAligment
from utils.file_helper import FileHelper
import cv2

s_cnt = 0

def do_trace(filenames, plot=True, gender=None):
    global s_cnt
    print()
    mem_dump("ck01")
    ImgpFaceAligment.init_imgp()
    mem_dump("ck02")

    for filename in filenames:
        if not filename.startswith("/"):
            filename = dir_root + os.sep + filename

        img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR)
        img_resized = cv2.resize(img, (512, 512))
        img_aligned, fa_ret = ImgpFaceAligment.make_aligment(img_resized)

        mem_dump("ck10")

        del img
        del img_resized
        del img_aligned
        del fa_ret

        s_cnt += 1

        mem_dump("ck11")
        gc.collect()        
        mem_dump("ck12")

    mem_dump("ck03")
    ImgpFaceAligment.close_imgp()
    mem_dump("ck04")
    gc.collect()
    mem_dump("ck05")
    print()

    if plot:
        msg = "ImgpFaceAligment total file processed {} ".format(s_cnt)
        print(msg)
        plot_mem_history(title=msg)


if __name__ == '__main__':
    if __file__.startswith("/"):
        dir_root = os.path.dirname(os.path.dirname(__file__))
    else:
        dir_root = os.getcwd()

    for i in range(10):

        filenames = FileHelper.find_all_images(dir_root + os.sep + "dataset_org_hair_styles/Version 1.4/00")
        do_trace(filenames, plot=False, gender="M")

        filenames = FileHelper.find_all_images(dir_root + os.sep + "dataset_org_hair_styles/Version 1.4/01")
        do_trace(filenames, plot=False, gender="F")

        filenames = FileHelper.find_all_images(dir_root + os.sep + "dataset_org_hair_styles/Version 1.4/02")
        do_trace(filenames, plot=False, gender="F")

        filenames = SampleImages.get_sample_images_female()
        do_trace(filenames, plot=False)

        filenames = SampleImages.get_sample_images_male()
        do_trace(filenames, plot=False)

        filenames = SampleImages.get_sample_images_brd()
        do_trace(filenames, plot=False)

        filenames = SampleImages.get_sample_images_ctn()
        do_trace(filenames, plot=False)

        filenames = SampleImages.get_sample_images_icl()
        do_trace(filenames, plot=False)

    filenames = SampleImages.get_sample_images_hsi()
    do_trace(filenames)
