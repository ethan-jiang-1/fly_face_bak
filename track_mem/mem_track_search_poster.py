import os 
import sys
import gc
#import numpy as np

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from track_mem.mem_tracker import mem_dump, plot_mem_history
from utils_inspect.sample_images import SampleImages
from imgp_search_matched_poster import SearchMatchedPoster
from utils.file_helper import FileHelper

s_cnt = 0

def do_trace(filenames, plot=True, gender=None):
    global s_cnt
    print()
    mem_dump("ck01")
    smg = SearchMatchedPoster()
    smg.init_searcher()

    for filename in filenames:
        if not filename.startswith("/"):
            filename = dir_root + os.sep + filename

        mem_dump("ck10")
        smp_ret = smg.search_for_poster_with_gender(filename, gender=gender)
        #ffg.show_results(filename)

        s_cnt += 1

        mem_dump("ck11")
        del smp_ret
        mem_dump("ck12")
        gc.collect()        
        mem_dump("ck13")

    mem_dump("ck02")
    del smg
    mem_dump("ck03")
    gc.collect()
    mem_dump("ck04")
    print()

    if plot:
        msg = "total file processed {} ".format(s_cnt)
        print(msg)
        plot_mem_history(title=msg)


if __name__ == '__main__':
    if __file__.startswith("/"):
        dir_root = os.path.dirname(os.path.dirname(__file__))
    else:
        dir_root = os.getcwd()

    for i in range(50):

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
