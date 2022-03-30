import os
import sys 
#import cv2
from collections import namedtuple

BER_RESULT = namedtuple('BER_RESULT', "img_beard img_eyebrow") 

class ImgpBeardEyebrow():
    hsi_klass = None 

    @classmethod
    def init_imgp(cls):
        print("ImgpBeardEyebrow inited")

    @classmethod
    def close_imgp(cls):
        print("ImgpBeardEyebrow closed")

    @classmethod
    def extract_beard_eyebrow(cls, img_selfie, img_hair, mesh_results):
        img_beard = cls._locate_beard(img_selfie, mesh_results)
        img_eyebrow = cls._locate_eyebrow(img_selfie, mesh_results)
        return img_beard, img_eyebrow

    @classmethod
    def _locate_beard(cls, image, mesh_results):
        from fmx_mesh_beard import FmxMeshBeard
        return FmxMeshBeard.process_img(image, mesh_results)

    @classmethod
    def _locate_eyebrow(cls, image, mesh_results):
        from fmx_mesh_beard import FmxMeshBeard
        return FmxMeshBeard.process_img(image, mesh_results)

def do_exp():
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]

    from face_feature_generator import FaceFeatureGenerator
    from imgp_common import FileHelper

    ffg = FaceFeatureGenerator()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   

    filenames = FileHelper.find_all_images(src_dir)
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue

        ffg.process(filename) 

    del ffg 

if __name__ == '__main__':
    do_exp()
