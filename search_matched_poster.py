
from collections import namedtuple
from pprint import pprint

from clf_net.clf_face import ClfFace
from clf_net.clf_bread import ClfBeard
from clf_net.clf_hair import ClfHair
from bld_gen.poster_query import PosterQuery

from utils.colorstr import log_colorstr

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
        cls.init_searcher()

        imgs, names, title, dt = cls.ffg.process_image(img_filename)
        log_colorstr("blue", "#SMP: process {} as {} in {}secs".format(img_filename, title, dt.total_seconds()))

        img_org, img_hair, img_face, img_beard = cls.get_bin_images(imgs, names)

        hair_id = ClfHair.get_category_id(img_hair)
        beard_id = ClfBeard.get_category_id(img_beard)
        face_id = ClfFace.get_category_id(img_face)
        log_colorstr("blue", "#SMP: hair_id:{},  beard_id:{}, face_id:{}".format(hair_id, beard_id, face_id))

        poster_pathname, _ = PosterQuery.get_poster(hair_id, beard_id, face_id)
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

def do_exp():
    import cv2
    from utils.plot_helper import PlotHelper

    img_filename = "utils_inspect/_sample_imgs/brd_image2.jpeg"

    smp_ret = SearchMatchedPoster.search_for_poster(img_filename)
    pprint(smp_ret)

    img_poster = cv2.imread(smp_ret.poster_pathname)
    imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_face, smp_ret.img_beard, img_poster]
    names = ["org", "hair", "face", "beard", "poster"]

    PlotHelper.plot_imgs(imgs, names)

if __name__ == '__main__':
    do_exp()
