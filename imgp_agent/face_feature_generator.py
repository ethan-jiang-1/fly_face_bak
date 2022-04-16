
import os
import cv2
import sys 
from collections import namedtuple
from datetime import datetime

dir_this = os.path.dirname(__file__)
if dir_this not in sys.path:
    sys.path.append(dir_this)

from imgp_face_aligment import ImgpFaceAligment
from imgp_mp_selfie_marker import ImgpSelfieMarker
from imgp_mp_hair_marker import ImgpHairMarker
from imgp_mp_facemesh_extractor import ImgpFacemeshExtractor
from imgp_cv_beard_extractor import ImgpCvBeardExtractor
from imgp_common import FileHelper, PlotHelper

FFG_RESULT = namedtuple("FFG_RESULT", "imgs etnames err_code last_err")

class FaceFeatureGenerator(object):
    def __init__(self, debug=False):
        self.debug = debug
        ImgpFaceAligment.init_imgp()
        ImgpSelfieMarker.init_imgp()
        ImgpHairMarker.init_imgp()
        ImgpFacemeshExtractor.init_imgp()
        ImgpCvBeardExtractor.init_imgp()

        self.err_code = None
        self.last_err = None

    def __del__(self):
        ImgpFaceAligment.close_imgp()
        ImgpSelfieMarker.close_imgp()
        ImgpHairMarker.close_imgp()
        ImgpFacemeshExtractor.close_imgp()
        ImgpCvBeardExtractor.close_imgp()

    def show_results(self, filename,  selected_img_types=None):
        ffg_ret = self.process_image(filename)
        imgs, etnames = ffg_ret.imgs, ffg_ret.etnames

        imgs, etnames = self._filter_selected_types(imgs, etnames, selected_img_types)

        PlotHelper.plot_imgs_grid(imgs, etnames, title=os.path.basename(filename))  

        del ffg_ret

    def save_results(self, filename, output_dir=None, selected_img_types=None):
        ffg_ret = self.process_image(filename)
        imgs, etnames = ffg_ret.imgs, ffg_ret.etnames

        if output_dir is None:
            output_dir = os.path.dirname(os.path.dirname(__file__)) + os.sep + "_reserved_output_feature_gen"
            os.makedirs(output_dir, exist_ok=True)

        imgs, etnames = self._filter_selected_types(imgs, etnames, selected_img_types)

        fname = os.path.basename(filename).split(".")[0]
        for img, name in zip(imgs, etnames):
            if img is None:
                continue 
            if selected_img_types is not None:
                if name not in selected_img_types:
                    continue
            filename = "{}{}{}_{}.jpg".format(output_dir, os.sep, fname, name)
            cv2.imwrite(filename, img)
            print("{} saved".format(filename))

        del ffg_ret

    def _filter_selected_types(self, imgs, names, selected_img_types):
        if selected_img_types is None:
            return imgs, names 
        new_imgs = []
        new_names = []
        for img, name in zip(imgs, names):
            if name in selected_img_types:
                new_imgs.append(img)
                new_names.append(name)
        return new_imgs, new_names

    def _prompt_error(self, err_code,  error_msg):
        from utils.colorstr import log_colorstr
        self.err_code = err_code
        self.last_err = error_msg
        log_colorstr("red", self.last_err)

    def _load_image(self, filename):   
        if not os.path.isfile(filename):
            self._prompt_error("FE01", "ERROR(FE01): failed to locate image file {}".format(filename))
            return None

        img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img_org is None:
            if not cv2.haveImageReader(filename):
                self._prompt_error("FE02", "ERROR(FE02): failed to load image file {} as there no proper image reader to handle the format".format(filename))
            else:
                self._prompt_error("FE03", "ERROR(EF03): failed to load image file {}".format(filename))
            return None

        return img_org

    def _is_valid_face_image(self, fa_ret, filename):
        if fa_ret is None:
            self._prompt_error("FE10", "ERROR(FE10): failed to find alignment info file {}".format(filename))
            return False

        if fa_ret.img_ratoted_unified is None:
            self._prompt_error("FE11", "ERROR(FE11), failed to make aligment of the image file {}".format(filename))
            return False
        return True

    def process_image(self, filename, clean_mem=True):
        from utils.colorstr import log_colorstr

        self._reset_error()

        d0 = datetime.now()
        img_org = self._load_image(filename)
        if img_org is None:
            return FFG_RESULT(imgs=None, etnames=None, err_code=self.err_code, last_err=self.last_err)

        img_aligned, fa_ret = ImgpFaceAligment.make_aligment(img_org, debug=self.debug)
        if not self._is_valid_face_image(fa_ret, filename):
            return FFG_RESULT(imgs=None, etnames=None, err_code=self.err_code, last_err=self.last_err)

        img_selfie, img_selfie_mask = ImgpSelfieMarker.fliter_selfie(img_aligned, debug=self.debug)
        img_facemesh, fme_result = ImgpFacemeshExtractor.extract_mesh_features(img_selfie, debug=self.debug)
        img_hair, hsi_result = ImgpHairMarker.mark_hair(img_aligned, img_selfie_mask, debug=self.debug)
        img_beard, ber_result = ImgpCvBeardExtractor.extract_beard(img_aligned, img_selfie_mask, hsi_result.mask_black_sharp, fme_result.mesh_results, debug=self.debug)

        dt = datetime.now() - d0
        basename = os.path.basename(filename)
        log_colorstr("blue", "total inference time: for {}: {:.3f}sec\n".format(basename, dt.total_seconds()))

        imgs = [img_org, img_aligned, img_selfie,  img_facemesh, fme_result.img_facepaint, fme_result.img_outline, img_beard, img_hair]
        etnames = ["org", "aligned", "selfie", "facemesh", "facepaint", "outline", "beard", "hair"]

        if clean_mem:
            del img_selfie
            del img_selfie_mask
            del img_facemesh
            del fme_result
            del img_hair
            del hsi_result
            del img_beard
            del ber_result

        ffg_ret = FFG_RESULT(imgs=imgs,
                             etnames=etnames,
                             err_code=self.err_code, 
                             last_err=self.last_err)
        return ffg_ret

    def _reset_error(self):
        self.last_err = None
        self.err_code = None

    def get_last_error(self):
        return self.err_code, self.last_err

    def find_img_by_etname(self, imgs, etnames, expected_etname):
        if imgs is not None and etnames is not None:
            for idx, name in enumerate(etnames):
                if name == expected_etname:
                    return imgs[idx]
        return None

    def process_image_for_fename(self, filename, expected_etname):
        ffg_ret = self.process_image(filename)
        imgs, etnames = ffg_ret.imgs, ffg_ret.etnames
        if imgs is None:
            return None
        return self.find_img_by_etname(imgs, etnames, expected_etname)


def do_exp_file():
    #filename = "data/dataset_org_hair_styles/Version 1.1/02/001.jpeg"
    #filename = "_reserved_bug_imgs/hair_WechatIMG303.jpeg"

    #filename = "data/dataset_org_hair_styles/Version 1.4/00/00-002.jpg"
    #filename = "data/dataset_org_hair_styles/Version 1.4/00/00-003.jpg"
    filename = "data/dataset_org_hair_styles/Version 1.4/00/00-004.jpg"
    #filename = "data/dataset_org_hair_styles/Version 1.4/00/00-005.jpg"

    #filename = "data/dataset_org_hair_styles/Version 1.4/00/00-015.jpg"

    ffg = FaceFeatureGenerator(debug=True)
    ffg.show_results(filename) 
    del ffg 


def do_exp_dir():
    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]

    ffg = FaceFeatureGenerator()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    #src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   
    src_dir = os.sep.join([parent_dir, "data/dataset_org_hair_styles/Version 1.4/00"])   

    filenames = FileHelper.find_all_images(src_dir)
    filenames = sorted(filenames)
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue

        ffg.show_results(filename) 
        ffg.save_results(filename) 

    del ffg 


if __name__ == '__main__':
    #do_exp_file()
    do_exp_dir()
