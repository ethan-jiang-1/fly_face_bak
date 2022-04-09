import os
import sys 
import cv2
import numpy as np
from collections import namedtuple

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
            print("ImgpCvBeardExtractor inited")

    @classmethod
    def close_imgp(cls):
        cls.hsi_reference -= 1
        if cls.hsi_reference == 0:
            print("ImgpCvBeardExtractor closed")

    @classmethod
    def extract_beard(cls, img_aligned, img_selfie_mask, img_hair_black, mesh_results, debug=False):
        if BER_MODE == "GABOR":
            img_selfie_gray = cls._apply_gray_selfie_mask(img_aligned, img_selfie_mask)
            img_selfie_without_hair = cls._remove_hair(img_selfie_gray, img_hair_black, mesh_results, smart=True, debug=debug) 
            img_selfie_wb_no_hair = cls._white_balance_face(img_selfie_without_hair, debug=debug) 

            img_beard = cls._locate_beard_gabor(img_selfie_wb_no_hair, mesh_results, debug=debug)
        else:
            img_selfie_gray = cls._apply_gray_selfie_mask(img_aligned, img_selfie_mask)
            img_selfie_without_hair = cls._remove_hair(img_aligned, img_hair_black, mesh_results, smart=True, debug=debug) 
            img_selfie_wb_no_hair = cls._white_balance_face(img_selfie_without_hair, debug=debug) 

            img_beard = cls._locate_beard_otsu(img_selfie_wb_no_hair, mesh_results, debug=debug)

        if img_beard is not None:
            if img_beard.max() == img_beard.min():
                img_beard[:] = (0)
        ber_result = BER_RESULT(img_beard=img_beard)
        return img_beard, ber_result

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
        # 读取图像
        r, g, b = cv2.split(img_selfie)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        if r_avg != 0.0 and g_avg != 0.0 and b_avg != 0.0:
            # 求各个通道所占增益
            k = (r_avg + g_avg + b_avg) / 3
            kr = k / r_avg
            kg = k / g_avg
            kb = k / b_avg
            r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
            g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
            b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        img_selfie_wb = cv2.merge([b, g, r])
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([img_selfie, img_selfie_wb], names=["img", "white_banlanced"])
        return img_selfie_wb

    @classmethod
    def _locate_beard_gabor(cls, image, mesh_results, debug=False):
        try:
            from fcx_hair.fcx_beard_gabor import FcxBeardGabor
        except:
            from .fcx_hair.fcx_beard_gabor import FcxBeardGabor
        return FcxBeardGabor.process_img(image, mesh_results, debug=debug)

    @classmethod
    def _locate_beard_otsu(cls, image, mesh_results, debug=False):
        try:
            from fcx_hair.fcx_beard_otsu import FcxBeardOtsu
        except:
            from .fcx_hair.fcx_beard_otsu import FcxBeardOtsu
        return FcxBeardOtsu.process_img(image, mesh_results, debug=debug)

def do_exp_folder():
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
    selected_names = ["ctn_cartoon3.jpeg"]
    #selected_names = ["brd_image1.jpeg"]

    from face_feature_generator import FaceFeatureGenerator
    from imgp_common import FileHelper

    ffg = FaceFeatureGenerator()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    #src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   
    src_dir = os.sep.join([parent_dir, "dataset_org_beard_styles", "Beard Version 1.1", "03"])    

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

def do_exp_single():
    from face_feature_generator import FaceFeatureGenerator

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    #filename = "dataset_org_beard_styles/Beard Version 1.1/03/03_020.jpg"
    filename = "dataset_org_beard_styles/Beard Version 1.1/03/03_019.jpg"
    #filename = "dataset_org_beard_styles/Beard Version 1.1/03/03_018.jpg"

    full_path = parent_dir + os.sep + filename

    ffg = FaceFeatureGenerator(debug=True)

    ffg.show_results(full_path) 

    del ffg   

if __name__ == '__main__':
    #do_exp_folder()
    do_exp_single()
