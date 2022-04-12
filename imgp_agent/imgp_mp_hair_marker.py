#from datetime import datetime

import os
import sys
import cv2
from datetime import datetime

class ImgpHairMarker():
    hsi_klass = None 
    hsi_reference = 0
    hsi_instance = None

    @classmethod
    def init_imgp(cls):
        cls.hsi_reference += 1
        if cls.hsi_klass is None:
            if not cls.has_tflite_custom_installed():
                print("please go to hsi_tflite_interpeter foloder to install customized tflite-runtime-hsi")
                raise ValueError("no tflite-runtime-hsi installed")
            
            dir_parent = os.path.dirname(os.path.dirname(__file__))
            if dir_parent not in sys.path:
                sys.path.append(dir_parent)

            from imgp_agent.tflite_hair_segmentation import hair_segmentation_interpeter
            cls.hsi_klass = hair_segmentation_interpeter.HairSegmentationInterpreter
            print("ImgpHairMarker inited")
            cls.hsi_instance = cls.hsi_klass()

    @classmethod
    def close_imgp(cls):
        cls.hsi_reference -= 1
        if cls.hsi_reference == 0:
            if cls.hsi_klass is not None:
                cls.hsi_klass = None
                cls.hsi_instance = None
            print("ImgpHairMarker closed")

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
    def mark_hair(cls, image, debug=False):
        if cls.hsi_klass is None:
            cls.init_imgp()
        
        d0 = datetime.now()
        img_cv512 = cv2.resize(image, (512, 512))
        hsi_result = cls.hsi_instance.process_img_cv512(img_cv512)
        dt = datetime.now() - d0
        print("inference time(hm) :{:.3f}".format(dt.total_seconds()))
        return hsi_result.mask_white_sharp, hsi_result


def _mark_hair_imgs(src_dir):
    from imgp_agent.imgp_common import FileHelper
    filenames = FileHelper.find_all_images(src_dir)

    ImgpHairMarker.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, _ = ImgpHairMarker.mark_hair(image)
        if image is None:
            print("not able to mark hair on", filename)

        FileHelper.save_output_image(image, src_dir, filename, "hair")

    ImgpHairMarker.close_imgp()

def _mark_hair_img(src_filename):
    from imgp_common import PlotHelper
    ImgpHairMarker.init_imgp()
    image = cv2.imread(src_filename, cv2.IMREAD_COLOR)

    image_hair, _ = ImgpHairMarker.mark_hair(image)
    if image_hair is None:
        print("not able to mark hair on", src_filename)
    
    PlotHelper.plot_imgs([image, image_hair])
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

    #src_filename = "dataset_org_hair_styles/Version 1.1/02/001.jpeg"
    #_mark_hair_img(src_filename)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
