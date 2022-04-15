import os 
import sys
import gc

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from track_mem.mem_tracker import mem_dump
from utils_inspect.sample_images import SampleImages
from imgp_agent.face_feature_generator import FaceFeatureGenerator


def do_trace(filenames):
    print()
    mem_dump("ck01")
    ffg = FaceFeatureGenerator()

    for filename in filenames:
        if not filename.startswith("/"):
            filename = dir_root + os.sep + filename

        ffg.save_results(filename)
        #ffg.show_results(filename)

    mem_dump("ck02")
    del ffg
    mem_dump("ck03")
    gc.collect()
    mem_dump("ck04")
    print()


if __name__ == '__main__':
    filenames = SampleImages.get_sample_images_female()
    do_trace(filenames)

    filenames = SampleImages.get_sample_images_male()
    do_trace(filenames)
