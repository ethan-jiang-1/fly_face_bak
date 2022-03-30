
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
from imgp_cv_beard_eyebrow import ImgpBeardEyebrow
from imgp_common import FileHelper, PlotHelper

FFA_IMGS = namedtuple('FFA_IMGS', 'img_name img_org img_aligned img_selfie img_facemesh img_hair') 

class FaceFeatureGenerator(object):
    def __init__(self, debug=False):
        self.debug = debug
        ImgpFaceAligment.init_imgp()
        ImgpSelfieMarker.init_imgp()
        ImgpHairMarker.init_imgp()
        ImgpFacemeshExtractor.init_imgp()
        ImgpBeardEyebrow.init_imgp()

    def __del__(self):
        ImgpFaceAligment.close_imgp()
        ImgpSelfieMarker.close_imgp()
        ImgpHairMarker.close_imgp()
        ImgpFacemeshExtractor.close_imgp()
        ImgpBeardEyebrow.close_imgp()

    def process(self, filename, show=True):
        if not os.path.isfile(filename):
            return None, None, None, None 

        d0 = datetime.now()
        img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img_org is None:
            if not cv2.haveImageReader(filename):
                print("ERROR: failed to load image file {} as there no proper image reader to handle the format".format(filename))
            else:
                print("ERROR: failed to load image file {}".format(filename))
            return None, None, None, None

        img_aligned, fa_ret = ImgpFaceAligment.make_aligment(img_org)
        img_selfie, img_selfie_mask = ImgpSelfieMarker.fliter_selfie(img_aligned)
        img_facemesh, fme_result = ImgpFacemeshExtractor.extract_mesh_features(img_selfie)
        img_hair, _ = ImgpHairMarker.mark_hair(img_aligned)
        img_beard, img_eyebrow = ImgpBeardEyebrow.extract_beard_eyebrow(img_selfie, img_hair, fme_result.mesh_results)

        dt = datetime.now() - d0
        basename = os.path.basename(filename)
        print("total inference time: for {}: {:.3f}".format(basename, dt.total_seconds()))

        imgs = [img_org, img_aligned, img_selfie, img_selfie_mask, img_facemesh, fme_result.img_facepaint, img_beard, img_hair]
        names = ["org", "aligned", "selfie", "selfie_mask", "facemesh", "facepaint", "beard", "hair"]
        title = os.path.basename(filename)
        if show:
            PlotHelper.plot_imgs_grid(imgs, names, title=basename)
        return imgs, names, title, dt

def do_exp():
    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]

    ffg = FaceFeatureGenerator()

    parent_dir = os.path.dirname(os.path.dirname(__file__))
    src_dir = os.sep.join([parent_dir, "utils_inspect", "_sample_imgs"])   

    filenames = FileHelper.find_all_images(src_dir)
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue

        ffg.process(filename) 

    del ffg 

if __name__ == '__main__':
    do_exp()
