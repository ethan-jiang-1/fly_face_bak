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
        return hsi_ret.mask_white_sharp, None

def _find_all_images(src_dir):
    if not os.path.isdir(src_dir):
        raise ValueError("{} not folder".format(src_dir))

    names = os.listdir(src_dir)
    filenames = []
    for name in names:
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            filenames.append(os.sep.join([src_dir, name]))
    return filenames

def _mark_hair_imgs(src_dir):
    filenames = _find_all_images(src_dir)

    ImgpHairMarker.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, _ = ImgpHairMarker.mark_hair(image)
        if image is None:
            print("not able to mark landmark on", filename)

        #img_flip = cv2.flip(image, 1)
        output_dir = "_reserved_output_{}".format(os.path.basename(src_dir))
        os.makedirs(output_dir, exist_ok=True)
        dst_pathname = "{}{}{}".format(output_dir, os.sep, os.path.basename(filename).replace(".jp", "_ha.jp"))
        cv2.imwrite(dst_pathname, image)
        print("{} saved".format(dst_pathname))

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
