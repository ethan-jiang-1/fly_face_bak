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


def do_trace(filenames, plot=True):
    print()
    mem_dump("ck01")
    smg = SearchMatchedPoster()
    smg.init_searcher()

    for filename in filenames:
        if not filename.startswith("/"):
            filename = dir_root + os.sep + filename

        smp_ret = smg.search_for_poster(filename)
        #ffg.show_results(filename)

        del smp_ret

    mem_dump("ck02")
    del smg
    mem_dump("ck03")
    gc.collect()
    mem_dump("ck04")
    print()

    if plot:
        plot_mem_history()


if __name__ == '__main__':
    filenames = SampleImages.get_sample_images_female()
    do_trace(filenames, plot=False)

    filenames = SampleImages.get_sample_images_male()
    do_trace(filenames, plot=False)

    filenames = SampleImages.get_sample_images_brd()
    do_trace(filenames, plot=False)

    filenames = SampleImages.get_sample_images_hsi()
    do_trace(filenames)
