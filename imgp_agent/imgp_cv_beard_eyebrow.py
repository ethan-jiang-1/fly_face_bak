import os
import sys 
import cv2
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
    def extract_beard_eyebrow(cls, img_selfie, img_hair_black, mesh_results, debug=False):
        img_selfie_without_hair = cls._remove_hair(img_selfie, img_hair_black, debug=debug)  

        img_beard = cls._locate_beard(img_selfie_without_hair, mesh_results, debug=debug)
        img_eyebrow = cls._locate_eyebrow(img_selfie_without_hair, mesh_results, debug=debug)
        return img_beard, img_eyebrow

    @classmethod
    def _remove_hair(cls, img_selfie, img_hair_black, debug=False):
        # img_hair_mask = cv2.cvtColor(img_hair, cv2.COLOR_BGR2GRAY)
        img_selfie_without_hair = cv2.bitwise_and(img_selfie, img_selfie, mask = img_hair_black)
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([img_selfie, img_hair_black, img_selfie_without_hair])
        return img_selfie_without_hair

    @classmethod
    def _locate_beard(cls, image, mesh_results, debug=False):
        from fmx_mesh_beard import FmxMeshBeard
        return FmxMeshBeard.process_img(image, mesh_results, debug=debug)

    @classmethod
    def _locate_eyebrow(cls, image, mesh_results, debug=False):
        from fmx_mesh_beard import FmxMeshBeard
        return FmxMeshBeard.process_img(image, mesh_results, debug=debug)

def do_exp():
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]
    #selected_names = ["icl_image5.jpeg"]
    #selected_names = ["ctn_cartoon2.jpeg"]

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

        ffg.show_results(filename) 

    del ffg 

if __name__ == '__main__':
    do_exp()
