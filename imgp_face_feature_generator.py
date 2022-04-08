import os

from imgp_agent.face_feature_generator import FaceFeatureGenerator
from imgp_agent.imgp_common import FileHelper

def _do_exp_on_dir(src_dir):
    ffg = FaceFeatureGenerator()

    filenames = FileHelper.find_all_images(src_dir)
    #print(filenames)

    for filename in filenames:
        ffg.save_results(filename)

    del ffg


def do_exp():
    dir_root = os.path.dirname(__file__)
    
    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    src_dir = os.sep.join([dir_root, "utils_inspect", "_sample_imgs"])

    _do_exp_on_dir(src_dir)

if __name__ == '__main__':
    do_exp()
