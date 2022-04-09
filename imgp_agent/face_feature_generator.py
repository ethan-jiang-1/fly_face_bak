
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

FFA_IMGS = namedtuple('FFA_IMGS', 'img_name img_org img_aligned img_selfie img_facemesh img_hair') 

class FaceFeatureGenerator(object):
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

    def show_results(self, filename,  selected_img_types=None):
        imgs, names = self.process_image(filename)

        imgs, names = self._filter_selected_types(imgs, names, selected_img_types)

        PlotHelper.plot_imgs_grid(imgs, names, title=os.path.basename(filename))  

    def save_results(self, filename, output_dir=None, selected_img_types=None):
        imgs, names = self.process_image(filename)
        if output_dir is None:
            output_dir = os.path.dirname(os.path.dirname(__file__)) + os.sep + "_reserved_output_feature_gen"
            os.makedirs(output_dir, exist_ok=True)

        imgs, names = self._filter_selected_types(imgs, names, selected_img_types)

        fname = os.path.basename(filename).split(".")[0]
        for img, name in zip(imgs, names):
            if img is None:
                continue 
            if selected_img_types is not None:
                if name not in selected_img_types:
                    continue
            filename = "{}{}{}_{}.jpg".format(output_dir, os.sep, fname, name)
            cv2.imwrite(filename, img)
            print("{} saved".format(filename))

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

    def process_image(self, filename):
        from utils.colorstr import log_colorstr
        if not os.path.isfile(filename):
            return None, None, None, None 

        d0 = datetime.now()
        img_org = cv2.imread(filename, cv2.IMREAD_COLOR)
        if img_org is None:
            if not cv2.haveImageReader(filename):
                log_colorstr("red", "ERROR: failed to load image file {} as there no proper image reader to handle the format".format(filename))
            else:
                log_colorstr("red", "ERROR: failed to load image file {}".format(filename))
            return None, None, None, None

        img_aligned, fa_ret = ImgpFaceAligment.make_aligment(img_org, debug=self.debug)
        img_selfie, img_selfie_mask = ImgpSelfieMarker.fliter_selfie(img_aligned, debug=self.debug)
        img_facemesh, fme_result = ImgpFacemeshExtractor.extract_mesh_features(img_selfie, debug=self.debug)
        img_hair, hsi_result = ImgpHairMarker.mark_hair(img_aligned, debug=self.debug)
        img_beard, _ = ImgpCvBeardExtractor.extract_beard(img_aligned, img_selfie_mask, hsi_result.mask_black_sharp, fme_result.mesh_results, debug=self.debug)

        (fa_ret, img_selfie_mask)

        dt = datetime.now() - d0
        basename = os.path.basename(filename)
        log_colorstr("blue", "total inference time: for {}: {:.3f}sec\n".format(basename, dt.total_seconds()))

        imgs = [img_org, img_aligned, img_selfie,  img_facemesh, fme_result.img_facepaint, fme_result.img_outline, img_beard, img_hair]
        names = ["org", "aligned", "selfie", "facemesh", "facepaint", "outline", "beard", "hair"]
        return imgs, names

    def find_img(self, imgs, names, expected_name):
        for idx, name in enumerate(names):
            if name == expected_name:
                return imgs[idx]
        return None

    def process_image_for(self, filename, expected_name):
        imgs, names = self.process_image(filename)
        if imgs is None:
            return None
        return self.find_img(imgs, names, expected_name)


def do_exp_file():
    filename = "dataset_org_hair_styles/Version 1.1/02/001.jpeg"

    ffg0 = FaceFeatureGenerator()
    del ffg0

    ffg = FaceFeatureGenerator()
    ffg.show_results(filename) 
    del ffg 


def do_exp_dir():
    selected_names = None
    #selected_names = ["hsi_image1.jpeg"]

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
    do_exp_file()
    #do_exp_dir()
