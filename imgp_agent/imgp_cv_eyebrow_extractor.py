import os
import sys 
#import cv2
#import numpy as np
#from collections import namedtuple

EBR_DEBUG = False

class ImgpCvEyebrowExtractor():
    hsi_klass = None 
    hsi_reference = 0

    @classmethod
    def init_imgp(cls):
        cls.hsi_reference += 1
        if cls.hsi_reference == 1:
            print("#ImgpCvEyebrowExtractor inited")

    @classmethod
    def close_imgp(cls):
        cls.hsi_reference -= 1
        if cls.hsi_reference == 0:
            print("#ImgpCvEyebrowExtractor closed")

    @classmethod
    def extract_eyebrow(cls, img_selfie, img_hair_black, mesh_results, debug=EBR_DEBUG):
        return None, None
 
def do_exp():
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]
    #selected_names = ["hsi_image3.jpeg"]
    #selected_names = ["hsi_image8.jpeg"]
    #selected_names = ["icl_image4.jpeg"]
    #selected_names = ["icl_image5.jpeg"]
    #selected_names = ["ctn_cartoon2.jpeg"]
    #selected_names = ["ctn_cartoon3.jpeg"]
    #selected_names = ["brd_image1.jpeg"]

    from face_feature_generator import FaceFeatureGenerator
    from imgp_common import FileHelper

    ffg = FaceFeatureGenerator()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   

    filenames = FileHelper.find_all_images(src_dir)
    filenames = sorted(filenames)
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue

        ffg.show_results(filename) 
        #ffg.save_results(filename) 

    del ffg 

if __name__ == '__main__':
    do_exp()
