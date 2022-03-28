
import os
import cv2 

from collections import namedtuple
from imgp_face_aligment import ImgpFaceAligment
from imgp_mp_selfie import ImgpSelfieMarker
from imgp_mp_hair import ImgpHairMarker
from imgp_mp_facemesh import ImgpFacemeshMarker
from imgp_common import FileHelper, PlotHelper

FFA_IMGS = namedtuple('FFA_IMGS', 'img_name img_org img_aligned img_selfie img_facemesh img_hair') 

class FaceFeatureGenerator(object):
    def __init__(self, debug=False):
        self.debug = debug
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
        if not os.path.isfile(filename):
            return None 

        img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
        img_aligned, _ = ImgpFaceAligment.make_aligment(img_org)
        img_selfie, _ = ImgpSelfieMarker.fliter_selfie(img_aligned)
        img_facemesh, _ = ImgpFacemeshMarker.mark_face_mesh(img_selfie)
        img_hair, _ = ImgpHairMarker.mark_hair(img_selfie)

        PlotHelper.plot_imgs([img_org, img_aligned, img_selfie, img_facemesh, img_hair],
                             names = ["org", "aligned", "selfie", "facemesh", "hair"])


def do_exp():
    ffg = FaceFeatureGenerator()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   

    filenames = FileHelper.find_all_images(src_dir)
    for filename in filenames:
        ffg.process(filename) 

    del ffg 

if __name__ == '__main__':
    do_exp()
