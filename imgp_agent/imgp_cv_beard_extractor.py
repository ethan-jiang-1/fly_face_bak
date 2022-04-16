import os
import sys 
import cv2
import numpy as np
from datetime import datetime
from collections import namedtuple

dir_this = os.path.dirname(__file__)
if dir_this not in sys.path:
    sys.path.append(dir_this)

BER_RESULT = namedtuple('BER_RESULT', "img_beard") 
BER_MODE = "GABOR"
SFM_BG_COLOR = (192, 192, 192)

class ImgpCvBeardExtractor():
    hsi_klass = None 
    hsi_reference = 0

    @classmethod
    def init_imgp(cls):
        cls.hsi_reference += 1
        if cls.hsi_reference == 1:
            print("#ImgpCvBeardExtractor inited")

    @classmethod
    def close_imgp(cls):
        cls.hsi_reference -= 1
        if cls.hsi_reference == 0:
            print("#ImgpCvBeardExtractor closed")

    @classmethod
    def extract_beard(cls, img_aligned, img_selfie_mask, img_hair_black, mesh_results, debug=False):
        d0 = datetime.now()
        img_selfie_gray = cls._apply_gray_selfie_mask(img_aligned, img_selfie_mask)
        img_selfie_without_hair = cls._remove_hair(img_selfie_gray, img_hair_black, mesh_results, smart=True, debug=debug) 
        img_selfie_wb_no_hair = cls._white_balance_face(img_selfie_without_hair, debug=debug) 

        if BER_MODE == "GABOR":
            img_beard = cls._locate_beard_gabor(img_selfie_wb_no_hair, mesh_results, debug=debug)
        elif BER_MODE == "OTSU":
            img_beard = cls._locate_beard_otsu(img_selfie_wb_no_hair, mesh_results, debug=debug)
        else:
            raise ValueError("BER_MODE: {} not supported".format(BER_MODE))

        ber_result = BER_RESULT(img_beard=img_beard)
        dt = datetime.now() - d0 
        print("inference time(eb) :{:.3f}".format(dt.total_seconds()))

        return img_beard, ber_result

    @classmethod
    def _revert_image(cls, img1):
        h,w,c=img1.shape
        img2=np.zeros((h,w,c),dtype=img1.dtype)
        invert_image=cv2.addWeighted(img1,-1,img2,0,255,0)
        return invert_image

    @classmethod
    def _apply_gray_selfie_mask(cls, img, img_selfie_mask):
        condition = np.stack((img_selfie_mask,) * 3 , axis=-1) > 127
        fg_image = img
        bg_image = np.zeros(img.shape, dtype=np.uint8)
        bg_image[:] = SFM_BG_COLOR 
        output_image = np.where(condition, fg_image, bg_image)
        return output_image

    @classmethod
    def _apply_gray_hair_mask(cls, img, img_hair_mask):
        condition = np.stack((img_hair_mask,) * 3 , axis=-1) > 127
        fg_image = img
        bg_image = np.zeros(img.shape, dtype=np.uint8)
        bg_image[:] = SFM_BG_COLOR 
        output_image = np.where(condition, fg_image, bg_image)
        return output_image

    @classmethod
    def _get_smart_hair_mask(cls, img_hair_black, mesh_results):
        img_hair_black_smart = np.zeros(img_hair_black.shape, dtype=img_hair_black.dtype)
        img_hair_black_smart += (192)
        h = img_hair_black.shape[0] 
        img_hair_black_smart[0:int(h/2),:] = img_hair_black[0:int(h/2),:]   
        #img_hair_black_smart[int(h/2):h,:] = img_hair_black[int(h/2):h,:]   
        return img_hair_black_smart

    @classmethod
    def _remove_hair(cls, img_selfie, img_hair_black, mesh_results, smart=True, debug=False):
        if img_hair_black.max() == 0:
            img_selfie_without_hair = img_selfie  # no hair to remove
        else:
            if smart:
                img_hair_black_smart = cls._get_smart_hair_mask(img_hair_black, mesh_results)
                img_selfie_without_hair = cls._apply_gray_hair_mask(img_selfie, img_hair_black_smart)
                #img_selfie_without_hair = cv2.bitwise_and(img_selfie, img_selfie, mask = img_hair_black_upper)
                if debug:
                    from imgp_common import PlotHelper
                    PlotHelper.plot_imgs([img_selfie, img_hair_black, img_hair_black_smart, img_selfie_without_hair], names=["selfie", "hair_black", "hair_black_smart", "selfie_without_hair"])
            else:
                img_selfie_without_hair = cls._apply_gray_hair_mask(img_selfie, img_hair_black)
                #img_selfie_without_hair = cv2.bitwise_and(img_selfie, img_selfie, mask = img_hair_black)
                if debug:
                    from imgp_common import PlotHelper
                    PlotHelper.plot_imgs([img_selfie, img_hair_black, img_selfie_without_hair], names=["selfie", "hair_black", "selfie_without_hair"])
        return img_selfie_without_hair

    @classmethod
    def _white_balance_face(cls, img_selfie, debug=False):
        from fcx_hair.fcx_base import FcxBase
        img_selfie_wb = FcxBase.ipc_white_balance_color(img_selfie)
        #if debug:
        #    from imgp_common import PlotHelper
        #    PlotHelper.plot_imgs([img_selfie, img_selfie_wb], names=["img", "white_banlanced"])
        return img_selfie_wb

    @classmethod
    def _locate_beard_gabor(cls, image, mesh_results, debug=False):
        from fcx_hair.fcx_beard_gabor import FcxBeardGabor
        return FcxBeardGabor.process_img(image, mesh_results, debug=debug)

    @classmethod
    def _locate_beard_otsu(cls, image, mesh_results, debug=False):
        from fcx_hair.fcx_beard_otsu import FcxBeardOtsu
        return FcxBeardOtsu.process_img(image, mesh_results, debug=debug)

def do_exp_folder(src_dir=None, quick_check=True):
    from face_feature_generator import FaceFeatureGenerator
    from imgp_common import FileHelper, PlotHelper

    parent_dir = os.path.dirname(os.path.dirname(__file__))

    if src_dir is None:
        #src_dir = "/Users/bowhead/prj_fly_face/data/dataset_org_beard_styles/Beard Version 1.1/01"
        #src_dir = "/Users/bowhead/prj_fly_face/data/dataset_org_beard_styles/Beard Version 1.1/02"
        #src_dir = "/Users/bowhead/prj_fly_face/data/dataset_org_beard_styles/Beard Version 1.1/03"
        src_dir = "utils_inspect/_sample_imgs"
    
    if not src_dir.startswith("/"):
        src_dir = parent_dir + "/" + src_dir

    ffg = FaceFeatureGenerator()

    filenames = FileHelper.find_all_images(src_dir)
    filenames = sorted(filenames)

    if quick_check:
        org_imgs = []
        brd_imgs = []
        for filename in filenames:
            ffg_ret = ffg.process_image(filename)
            imgs, etnames = ffg_ret.imgs, ffg_ret.etnames

            org_imgs.append(ffg.find_img_by_etname(imgs, etnames, "aligned"))
            brd_imgs.append(ffg.find_img_by_etname(imgs, etnames, "beard"))

        PlotHelper.plot_imgs_grid_2(org_imgs, brd_imgs, mod_num=6)
    else:
        for filename in filenames:
            ffg.show_results(filename) 

    del ffg

def do_exp_single(filename=None):
    from face_feature_generator import FaceFeatureGenerator

    parent_dir = os.path.dirname(os.path.dirname(__file__))

    if filename is None:
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/03/03_020.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/03/03_019.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/03/03_018.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/03/03_017.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/03/03_016.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/03/03_015.jpg"

        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_001.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_002.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_003.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_004.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_005.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_006.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_007.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_008.jpg"
        #filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_009.jpg"
        filename = "data/dataset_org_beard_styles/Beard Version 1.1/01/01_010.jpg"

    if not filename.startswith("/"):
        filename = parent_dir + os.sep + filename

    ffg = FaceFeatureGenerator(debug=True)

    ffg.show_results(filename) 

    del ffg   

if __name__ == '__main__':
    do_exp_folder()
    #do_exp_single()
