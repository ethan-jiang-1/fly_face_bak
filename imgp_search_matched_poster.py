#import os
#import PySimpleGUI as sg

from collections import namedtuple
#from pprint import pprint
from datetime import datetime
import numpy as np

from clf_net.clf_face import ClfFace
from clf_net.clf_beard import ClfBeard
from clf_net.clf_hair import ClfHair
from bld_gen.poster_query_local import PosterQueryLocal

from utils.colorstr import log_colorstr
import cv2

SMP_RESULT = namedtuple('SMP_RESULT', "img_org img_hair img_beard img_face hair_id beard_id face_id poster_pathname") 

class SearchMatchedPoster():
    ffg = None
    
    @classmethod
    def init_searcher(cls):
        if cls.ffg is None:
            from imgp_agent.face_feature_generator import FaceFeatureGenerator
            cls.ffg = FaceFeatureGenerator()

    @classmethod
    def close_searcher(cls):
        if cls.ffg:
            del cls.ffg 

    @classmethod
    def get_bin_images(cls, imgs, names):
        img_org = None
        img_hair = None 
        img_face = None
        img_beard = None
        for idx, name in enumerate(names):
            if name == "hair":
                img_hair = imgs[idx]
            elif name == "outline":
                img_face = imgs[idx]
            elif name == "beard":
                img_beard = imgs[idx] 
            elif name == "org":
                img_org = imgs[idx]
        return img_org, img_hair, img_face, img_beard

    @classmethod
    def search_for_poster(cls, img_filename):
        import os
        cls.init_searcher()

        imgs, etnames = cls.ffg.process_image(img_filename)
        title = os.path.basename(img_filename)
        log_colorstr("blue", "#SMP: process {} as {}".format(img_filename, title))

        img_org, img_hair, img_face, img_beard = cls.get_bin_images(imgs, etnames)

        gender = title.split('_')[0]
        if gender not in ['F', 'M']:
            log_colorstr("red", "#SMP: not able to guess gender from filename, should starts with F_ or M_: {}".format(img_filename))
            raise ValueError("filename should starts with F_ or M_: {}".format(title))

        hair_id = ClfHair.get_category_id(img_hair, gender=gender)
        beard_id = ClfBeard.get_category_id(img_beard, gender=gender)
        face_id = ClfFace.get_category_id(img_face, gender=gender)

        log_colorstr("blue", "#SMP: hair_id:{},  beard_id:{}, face_id:{}".format(hair_id, beard_id, face_id))

        poster_pathname, _ = PosterQueryLocal.get_poster(hair_id, beard_id, face_id, gender)
        log_colorstr("blue", "#SMP: poster_pathname: {}".format(poster_pathname))
        
        smp_ret = SMP_RESULT(img_org=img_org,
                             img_hair=img_hair,
                             img_beard=img_beard,
                             img_face=img_face,
                             hair_id=hair_id,
                             beard_id=beard_id,
                             face_id=face_id,
                             poster_pathname=poster_pathname)
        return smp_ret

def _debug_search_result(filename, smp_ret):
    from utils.plot_helper import PlotHelper
    img_poster = cv2.imread(smp_ret.poster_pathname)
    imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_beard, smp_ret.img_face, img_poster]
    names = ["org", "hair", "beard", "face", "poster"]

    PlotHelper.plot_imgs(imgs, names)

def _do_search_for_poster(img_filename, debug=False):
    log_colorstr("yellow", "from poster for  {}... ".format(img_filename))
    d0 = datetime.now()
    smp_ret = SearchMatchedPoster.search_for_poster(img_filename)
    # pprint(smp_ret, compact=True)
    dt = datetime.now() - d0
    print()
    log_colorstr("green", "Search poster that matches hair_id:{} beard_id:{} face_id:{}".format(smp_ret.hair_id, smp_ret.beard_id, smp_ret.face_id))
    log_colorstr("green", "Matched Poster: {}".format(smp_ret.poster_pathname))
    log_colorstr("yellow", "Done in {:.3f}sec".format(dt.total_seconds()))
    print()

    if debug:
        _debug_search_result(img_filename, smp_ret)

    return smp_ret, dt


def do_exp(debug):

    from utils.colorstr import log_colorstr

    SearchMatchedPoster.init_searcher()

    filenames = []
    filenames.append("utils_inspect/_sample_imgs/M_sun_guangtou.jpg")
    filenames.append("utils_inspect/_sample_imgs/M_sun_me.jpg")
    filenames.append("utils_inspect/_sample_imgs/F_sun_cancan.jpg")
    filenames.append("utils_inspect/_sample_imgs/F_sun_girl_1.jpg")

    dts = []
    for filename in filenames:
        _, dt = _do_search_for_poster(filename, debug=debug)
        dts.append(dt.total_seconds())

    np_dts = np.array(dts)
    print()
    log_colorstr("yellow", "avg seconds: {:.3f}sec".format(np_dts.mean()))
    log_colorstr("yellow", "min seconds: {:.3f}sec".format(np_dts.min()))
    log_colorstr("yellow", "max seconds: {:.3f}sec".format(np_dts.max()))
    print()


def do_exp_single(debug):
    from utils.colorstr import log_colorstr

    SearchMatchedPoster.init_searcher()

    filename = "utils_inspect/_sample_imgs/F_hsi_image8.jpeg"

    smp_ret, dt = _do_search_for_poster(filename, debug=debug)
    del smp_ret

    print()
    log_colorstr("yellow", "seconds: {:.3f}sec".format(dt.total_seconds()))
    print()

if __name__ == '__main__':
    #do_exp(debug=False)
    do_exp_single(debug=False)
