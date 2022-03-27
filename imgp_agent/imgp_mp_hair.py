#from datetime import datetime

import os
import cv2

class ImgpHairMarker():
    hsi_klass = None 

    @classmethod
    def init_imgp(cls):
        if cls.hsi_klass is not None:
            return 
    
        if not cls.has_tflite_custom_installed():
            print("please go to hsi_tflite_interpeter foloder to install customized tflite-runtime-hsi")
            raise ValueError("no tflite-runtime-hsi installed")
        from hsi_tflite_interpeter.tflite_hair_segmentation import hair_segmentation_interpeter
        cls.hsi_klass = hair_segmentation_interpeter.HairSegmentationInterpreter

    @classmethod
    def close_imgp(cls):
        if cls.hsi_klass is not None:
            cls.hsi_klass = None

    @classmethod
    def has_tflite_custom_installed(cls):
        from pkgutil import iter_modules
        for pmodule in iter_modules():
            if pmodule.name == "tflite_runtime":
                path = pmodule.module_finder.path
                if path.find("tflite_custom") != -1:
                    print(pmodule)
                    print(path)
                    return True
        return False

    @classmethod
    def mark_hair(cls, image):
        if cls.hsi_klass is None:
            cls.init_imgp()
        
        hsi = cls.hsi_klass()
        img_cv512 = cv2.resize(image, (512, 512))
        hsi_ret = hsi.process_img_cv512(img_cv512)
        return hsi_ret.mask_white_sharp, hsi_ret


def _mark_hair_imgs(src_dir):
    from imgp_agent.imgp_common import FileOp
    filenames = FileOp.find_all_images(src_dir)

    ImgpHairMarker.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, _ = ImgpHairMarker.mark_hair(image)
        if image is None:
            print("not able to mark hair on", filename)

        FileOp.save_output_image(image, src_dir, filename, "hair")

    ImgpHairMarker.close_imgp()

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
