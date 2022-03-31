import os
import sys 
import cv2
import numpy as np
from collections import namedtuple

BER_RESULT = namedtuple('BER_RESULT', "img_beard img_eyebrow") 
BER_DEBUG = False
BER_MODE = "new"

class ImgpBeardEyebrow():
    hsi_klass = None 

    @classmethod
    def init_imgp(cls):
        print("ImgpBeardEyebrow inited")

    @classmethod
    def close_imgp(cls):
        print("ImgpBeardEyebrow closed")

    @classmethod
    def extract_beard_eyebrow(cls, img_selfie, img_hair_black, mesh_results, debug=BER_DEBUG):
        if BER_MODE == "new":
            img_beard = cls._locate_beard_new(img_selfie, mesh_results, debug=debug)
            #img_eyebrow = cls._locate_eyebrow(img_selfie, mesh_results, debug=debug)
            img_eyebrow = None            
        else:
            img_selfie_without_hair = cls._remove_hair(img_selfie, img_hair_black, debug=debug) 
            img_selfie_wb_no_hair = cls._white_balance_face(img_selfie_without_hair, debug=debug) 

            img_beard = cls._locate_beard_old(img_selfie_wb_no_hair, mesh_results, debug=debug)
            #img_eyebrow = cls._locate_eyebrow(img_selfie, mesh_results, debug=debug)
            img_eyebrow = None

        ber_result = BER_RESULT(img_beard=img_beard,
                                img_eyebrow=img_eyebrow)
        return img_beard, img_eyebrow, ber_result

    @classmethod
    def _remove_hair(cls, img_selfie, img_hair_black, debug=False):
        if img_hair_black.max() == 0:
            img_selfie_without_hair = img_selfie  # no hair to remove
        else:
            img_hair_black_upper = np.zeros(img_hair_black.shape, dtype=img_hair_black.dtype)
            img_hair_black_upper += 255
            h = img_hair_black.shape[0] 
            img_hair_black_upper[0:int(h/2),:] = img_hair_black[0:int(h/2),:]
            img_selfie_without_hair = cv2.bitwise_and(img_selfie, img_selfie, mask = img_hair_black_upper)
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([img_selfie, img_hair_black, img_selfie_without_hair])
        return img_selfie_without_hair

    @classmethod
    def _white_balance_face(cls, img_selfie, debug=False):
        # 读取图像
        r, g, b = cv2.split(img_selfie)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]
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
            PlotHelper.plot_imgs([img_selfie, img_selfie_wb])
        return img_selfie

    @classmethod
    def _locate_beard_old(cls, image, mesh_results, debug=False):
        from fcx_hair.fcx_beard_old import FcxBeardOld
        return FcxBeardOld.process_img(image, mesh_results, debug=debug)

    @classmethod
    def _locate_beard_new(cls, image, mesh_results, debug=False):
        from fcx_hair.fcx_beard_new import FcxBeardNew
        return FcxBeardNew.process_img(image, mesh_results, debug=debug)

    @classmethod
    def _locate_eyebrow(cls, image, mesh_results, debug=False):
        from fcx_hair.fcx_eyebrow import FcxEyebrow
        return FcxEyebrow.process_img(image, mesh_results, debug=debug)

def do_exp():
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]
    #selected_names = ["hsi_image3.jpeg"]
    #selected_names = ["hsi_image8.jpeg"]
    #selected_names = ["icl_image5.jpeg"]
    #selected_names = ["ctn_cartoon2.jpeg"]
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
