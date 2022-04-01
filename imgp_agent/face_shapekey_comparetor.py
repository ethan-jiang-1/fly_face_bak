
import os
import cv2 
import sys
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


class FaceShapekeyComparetor(object):
    def __init__(self, debug=False):
        self.debug = debug
        ImgpFaceAligment.init_imgp()
        ImgpSelfieMarker.init_imgp()
        ImgpHairMarker.init_imgp()
        ImgpFacemeshExtractor.init_imgp()
        ImgpCvBeardExtractor.init_imgp()

    def __del__(self):
        ImgpFaceAligment.close_imgp()
        ImgpSelfieMarker.close_imgp()
        ImgpHairMarker.close_imgp()
        ImgpFacemeshExtractor.close_imgp()
        ImgpCvBeardExtractor.close_imgp()

    def process(self, filenames):
        from utils.colorstr import log_colorstr

        imgs = []
        names = []
        for filename in filenames:
            if not os.path.isfile(filename):
                return None 

            d0 = datetime.now()
            img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
            if img_org is None:
                if not cv2.haveImageReader(filename):
                    log_colorstr("red", "ERROR: failed to load image file {} as there no proper image reader to handle the format".format(filename))
                else:
                    log_colorstr("red", "ERROR: failed to load image file {}".format(filename))
                continue

            img_aligned, _ = ImgpFaceAligment.make_aligment(img_org)
            img_selfie, _ = ImgpSelfieMarker.fliter_selfie(img_aligned)
            img_facemesh, fme_result = ImgpFacemeshExtractor.extract_mesh_features(img_selfie)
            img_hair, hsi_result = ImgpHairMarker.mark_hair(img_aligned)
            img_beard, _ = ImgpCvBeardExtractor.extract_beard(img_selfie, hsi_result.mask_black_sharp, fme_result.mesh_results)

            dt = datetime.now() - d0
            log_colorstr("blue", "total inference time: for {}: {:.3f}".format(os.path.basename(filename), dt.total_seconds()))

            imgs.extend([img_org, img_facemesh, fme_result.img_facepaint, fme_result.img_outline, img_beard, img_hair])
            names.extend(["org", "facemesh", "facepaint", "outline", "beard", "hair"])

        PlotHelper.plot_imgs_grid(imgs, names, mod_num=6)


def do_exp():
    selected_names = None
    num_display = 2
    #selected_names = ["hsi_image1.jpeg"]

    fsc = FaceShapekeyComparetor()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   

    filenames = FileHelper.find_all_images(src_dir)
    filenames = sorted(filenames)
    sel_fnames = []
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue
        sel_fnames.append(filename)
        if len(sel_fnames) >= num_display:
            fsc.process(sel_fnames) 
            sel_fnames = []
    if len(sel_fnames) != 0:
        fsc.process(filenames) 

    del fsc 

if __name__ == '__main__':
    do_exp()
