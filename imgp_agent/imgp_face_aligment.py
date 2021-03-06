#from datetime import datetime

import os
import cv2
import sys

USING_FACE_ALIGMENT = "MP"  
#USING_FACE_ALIGMENT = "DLIB" 
#USING_FACE_ALIGMENT = "CV2"

class ImgpFaceAligment():
    fal_klass = None
    fal_reference = 0 

    @classmethod
    def init_imgp(cls):
        cls.fal_reference += 1
        if cls.fal_klass is None:
            cls.fal_klass = cls._pickup_face_klass()
            print("#ImgpFaceAligment: inited")           

    @classmethod
    def close_imgp(cls):
        cls.fal_reference -= 1
        if cls.fal_reference == 0:
            if cls.fal_klass is not None:
                cls.fal_klass = None
            print("#ImgpFaceAligment: closed")   

    @classmethod
    def _pickup_face_klass(cls):
        dir_parent = os.path.dirname(os.path.dirname(__file__))
        if dir_parent not in sys.path:
            sys.path.append(dir_parent)
        if USING_FACE_ALIGMENT == "MP": 
            from imgp_agent.face_alignment.face_aligment_mp import FaceAlignemtMp as FaceAligment  
        #elif USING_FACE_ALIGMENT == "DLIB":
        #    from imgp_agent.face_alignment.face_aligment_dlib import FaceAlignemtDlib as FaceAligment   
        #elif USING_FACE_ALIGMENT == "CV2":
        #    from imgp_agent.face_alignment.face_aligment_cv2 import FaceAlignemtCv2 as FaceAligment
        return FaceAligment  

    @classmethod
    def make_aligment(cls, image, debug=False):
        if cls.fal_klass is None:
            cls.init_imgp()
        
        fa = cls.fal_klass(debug=debug)
        fa.create_detector()

        fa_ret = fa.align_face(image)

        fa.close_detector()

        return fa_ret.img_ratoted_unified, fa_ret


def _mark_hair_imgs(src_dir):
    from imgp_agent.imgp_common import FileHelper
    filenames = FileHelper.find_all_images(src_dir)

    ImgpFaceAligment.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, _ = ImgpFaceAligment.make_aligment(image)
        if image is None:
            print("not able to mark hair on", filename)

        FileHelper.save_output_image(image, src_dir, filename, "aligment_{}".format(USING_FACE_ALIGMENT.lower()))

    ImgpFaceAligment.close_imgp()

def _get_root_dir():
    import os 

    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    return dir_root

def _add_root_in_sys_path():
    import sys 

    dir_root = _get_root_dir()
    if dir_root not in sys.path:
        sys.path.append(dir_root)

def do_exp():
    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    src_dir = os.sep.join([_get_root_dir(), "utils_inspect", "_sample_imgs"])
   
    _mark_hair_imgs(src_dir)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
