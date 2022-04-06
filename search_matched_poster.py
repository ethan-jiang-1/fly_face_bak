
from collections import namedtuple
from pprint import pprint

from clf_net.clf_face import ClfFace
from clf_net.clf_bread import ClfBeard
from clf_net.clf_hair import ClfHair
from poster_query.poster_query import PosterQuery

SMP_RESULT = namedtuple('SMP_RESULT', "img_org img_hair img_face img_beard hair_id face_id beard_id poster_pathname") 

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
        print(title, dt.total_seconds())

        img_org, img_hair, img_face, img_beard = cls.get_bin_images(imgs, names)

        hair_id = ClfHair.get_category_id(img_hair)
        face_id = ClfFace.get_category_id(img_face)
        beard_id = ClfBeard.get_category_id(img_beard)
        print(hair_id, face_id, beard_id)

        poster_pathname = PosterQuery.get_poster(hair_id, face_id, beard_id)
        print("poster_pathname", poster_pathname)
        
        smp_ret = SMP_RESULT(img_org=img_org,
                             img_hair=img_hair,
                             img_face=img_face,
                             img_beard=img_beard,
                             hair_id=hair_id,
                             face_id=face_id,
                             beard_id=beard_id,
                             poster_pathname=poster_pathname)
        return smp_ret

def do_exp():
    img_filename = "utils_inspect/_sample_imgs/brd_image2.jpeg"

    smp_ret = SearchMatchedPoster.search_for_poster(img_filename)
    pprint.pprint(smp_ret)

if __name__ == '__main__':
    do_exp()
