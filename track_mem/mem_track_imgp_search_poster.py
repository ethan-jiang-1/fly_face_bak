import os 
import sys
import gc

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from track_mem.mem_tracker import mem_dump
from utils_inspect.sample_images import SampleImages
from imgp_search_matched_poster import SearchMatchedPoster


def do_trace(filenames):
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


if __name__ == '__main__':
    filenames = SampleImages.get_sample_images_female()
    do_trace(filenames)

    filenames = SampleImages.get_sample_images_male()
    do_trace(filenames)
