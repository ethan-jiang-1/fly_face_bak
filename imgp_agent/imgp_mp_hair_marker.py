#from datetime import datetime

import os
import sys
import cv2
from datetime import datetime
import numpy as np

SFM_BG_COLOR = (192, 192, 192)

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
                    print("tflite_custom modele found", pmodule, "at", path)
                    return True
        return False

    @classmethod
    def mark_hair(cls, image, img_selfie_mask, debug=False):
        from imgp_agent.fcx_hair.fcx_base import FcxBase
        if cls.hsi_klass is None:
            cls.init_imgp()
        
        d0 = datetime.now()
        img_cv512 = cv2.resize(image, (512, 512))

        if img_selfie_mask is not None and img_selfie_mask.shape[0] == 512 and img_selfie_mask.shape[1] == 512:
            img_cv512 = cls._apply_white_selfie_mask(img_cv512, img_selfie_mask)

        img_wb = FcxBase.ipc_white_balance_color(img_cv512)
        hsi_result = cls.hsi_instance.process_img_cv512(img_wb)
        dt = datetime.now() - d0
        print("inference time(hm) :{:.3f}".format(dt.total_seconds()))
        if debug:
            from imgp_agent.imgp_common import PlotHelper
            PlotHelper.plot_imgs([image, img_selfie_mask, img_cv512, img_wb, hsi_result.mask_white_sharp],
                                 names=["org", "selfie_mask", "512_masked", "whitebanlanced", "hair"])
        return hsi_result.mask_white_sharp, hsi_result

    @classmethod
    def _apply_white_selfie_mask(cls, img, img_selfie_mask):
        condition = np.stack((img_selfie_mask,) * 3 , axis=-1) > 127
        fg_image = img
        bg_image = np.zeros(img.shape, dtype=np.uint8)
        bg_image[:] = (255, 255, 255) 
        output_image = np.where(condition, fg_image, bg_image)
        return output_image

def _get_selfie_mask(image):
    from imgp_agent.imgp_mp_selfie_marker import ImgpSelfieMarker
    ImgpSelfieMarker.init_imgp()
    img_cv512 = cv2.resize(image, (512, 512))
    _, img_selfie_mask  = ImgpSelfieMarker.fliter_selfie(img_cv512)
    return img_selfie_mask

def _mark_hair_imgs(src_dir, debug=False):
    from imgp_agent.imgp_common import FileHelper, PlotHelper
    filenames = FileHelper.find_all_images(src_dir)

    ImgpHairMarker.init_imgp()
    imgs = []
    names = []
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue
        
        selfie_mask = _get_selfie_mask(image)
        image, _ = ImgpHairMarker.mark_hair(image, selfie_mask, debug=False)
        if image is None:
            print("not able to mark hair on", filename)

        FileHelper.save_output_image(image, src_dir, filename, "hair")
        imgs.append(image)
        names.append(os.path.basename(filename).split(".")[0])

    if debug:
        PlotHelper.plot_imgs_grid(imgs, names, mod_num=5, set_axis_off=True)

    ImgpHairMarker.close_imgp()

def _mark_hair_img(src_filename, debug=False):
    from imgp_common import PlotHelper
    ImgpHairMarker.init_imgp()
    image = cv2.imread(src_filename, cv2.IMREAD_COLOR)

    selfie_mask = _get_selfie_mask(image)
    image_hair, _ = ImgpHairMarker.mark_hair(image, selfie_mask, debug=debug)
    if image_hair is None:
        print("not able to mark hair on", src_filename)
    
    PlotHelper.plot_imgs([image, selfie_mask, image_hair])
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

def do_exp_folder():
    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    #src_dir = os.sep.join([_get_root_dir(), "utils_inspect", "_sample_imgs"])
    src_dir = os.sep.join([_get_root_dir(), "dataset_org_hair_styles/Version 1.4/00"])

    _mark_hair_imgs(src_dir, debug=True)

    #src_filename = "dataset_org_hair_styles/Version 1.1/02/001.jpeg"
    #_mark_hair_img(src_filename)

def do_exp_single():
    dir_root = _get_root_dir()

    #filename = "dataset_org_hair_styles/Version 1.4/00/00-001.jpg"
    #filename = "dataset_org_hair_styles/Version 1.4/00/00-002.jpg"
    #filename = "dataset_org_hair_styles/Version 1.4/00/00-003.jpg"
    #filename = "dataset_org_hair_styles/Version 1.4/00/00-004.jpg"
    #filename = "dataset_org_hair_styles/Version 1.4/00/00-005.jpg"

    filename = "dataset_org_hair_styles/Version 1.4/00/00-015.jpg"

    filename = dir_root + os.sep + filename 
    _mark_hair_img(filename, debug=True)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp_folder()
    #do_exp_single()
