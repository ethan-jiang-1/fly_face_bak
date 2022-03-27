#from datetime import datetime

import os
import cv2

class ImgpFaceAligment():
    hsi_klass = None 

    @classmethod
    def init_imgp(cls):
        if cls.hsi_klass is not None:
            return 
        from face_alignment.face_aligment_mp import FaceAlignemtMp    
        cls.hsi_klass = FaceAlignemtMp

    @classmethod
    def close_imgp(cls):
        if cls.hsi_klass is not None:
            cls.hsi_klass = None

    @classmethod
    def make_aligment(cls, image):
        if cls.hsi_klass is None:
            cls.init_imgp()
        
        fa = cls.hsi_klass()
        fa.create_detector()

        fa_ret = fa.align_face(image)

        fa.close_detector()

        return fa_ret.img_ratoted_unified, fa_ret


def _mark_hair_imgs(src_dir):
    from imgp_agent.imgp_common import FileOp
    filenames = FileOp.find_all_images(src_dir)

    ImgpFaceAligment.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, _ = ImgpFaceAligment.make_aligment(image)
        if image is None:
            print("not able to mark hair on", filename)

        FileOp.save_output_image(image, src_dir, filename, "hair")

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
