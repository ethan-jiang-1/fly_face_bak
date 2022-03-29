
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
from imgp_common import FileHelper, PlotHelper


class FaceShapekeyComparetor(object):
    def __init__(self, debug=False):
        self.debug = debug
        ImgpFaceAligment.init_imgp()
        ImgpSelfieMarker.init_imgp()
        ImgpHairMarker.init_imgp()
        ImgpFacemeshExtractor.init_imgp()

    def __del__(self):
        ImgpFaceAligment.close_imgp()
        ImgpSelfieMarker.close_imgp()
        ImgpHairMarker.close_imgp()
        ImgpFacemeshExtractor.close_imgp()

    def process(self, filenames):

        imgs = []
        names = []
        for filename in filenames:
            if not os.path.isfile(filename):
                return None 

            d0 = datetime.now()
            img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
            img_aligned, _ = ImgpFaceAligment.make_aligment(img_org)
            img_selfie, img_selfie_mask = ImgpSelfieMarker.fliter_selfie(img_aligned)
            img_facemesh, fme_result = ImgpFacemeshExtractor.extract_mesh_features(img_selfie)
            img_hair, _ = ImgpHairMarker.mark_hair(img_aligned)
            dt = datetime.now() - d0
            print("total inference time: for {}: {:.3f}".format(os.path.basename(filename), dt.total_seconds()))

            imgs.extend([img_org, img_facemesh, fme_result.img_facepaint, img_hair])
            names.extend(["org", "facemesh", "facepaint", "hair"])

        PlotHelper.plot_imgs_grid(imgs, names, mod_num=4)


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
