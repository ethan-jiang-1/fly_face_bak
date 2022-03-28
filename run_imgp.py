import os

from imgp_agent.imgp_face_aligment import ImgpFaceAligment
from imgp_agent.imgp_mp_selfie import ImgpSelfieMarker
from imgp_agent.imgp_mp_hair import ImgpHairMarker
from imgp_agent.imgp_mp_facemesh import ImgpFacemeshMarker
from imgp_agent.imgp_common import FileHelper

class FaceFeatureGen(object):
    def __init__(self):
        ImgpFaceAligment.init_imgp()
        ImgpSelfieMarker.init_imgp()
        ImgpHairMarker.init_imgp()
        ImgpFacemeshMarker.init_imgp()

    def __del__(self):
        ImgpFaceAligment.close_imgp()
        ImgpSelfieMarker.close_imgp()
        ImgpHairMarker.close_imgp()
        ImgpFacemeshMarker.close_imgp()

    def process(self, filename):
        pass 


def _do_exp_on_dir(src_dir):
    ffg = FaceFeatureGen()

    filenames = FileHelper.find_all_images(src_dir)
    #print(filenames)

    for filename in filenames:
        ffg.process(filename)

    del ffg


def do_exp():
    dir_root = os.path.dirname(__file__)

    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    src_dir = os.sep.join([dir_root, "utils_inspect", "_sample_imgs"])

    _do_exp_on_dir(src_dir)

if __name__ == '__main__':
    do_exp()
