# import os
# import sys
import numpy as np

class ClfFace(object):
    @classmethod
    def get_category_id(cls,  img_bin_face):
        return 0 

def do_exp(img_pathname):
    # dir_this = os.path.dirname(__file__)
    # if dir_this not in sys.path:
    #     sys.path.append(dir_this)
    #
    # if not img_pathname.startswith("/"):
    #     dir_root = os.path.dirname(dir_this)
    #     img_pathname = "{}/{}".format(dir_root, img_pathname)
    #
    # id = ClfFace.get_category_id(img_pathname)
    id = np.random.randint(0,3)
    print("category id for {} is {}".format(img_pathname, id))
    return id

if __name__ == '__main__':
    filename = "../_reserved_output_feature_gen/01_001_hair.jpg"
    do_exp(filename)
