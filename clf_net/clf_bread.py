import os
import sys 


class ClfBeard(object):
    @classmethod
    def get_category_id(cls,  img_bin_beard):
        return 0 

def do_exp(img_pathname):
    dir_this = os.path.dirname(__file__)
    if dir_this not in sys.path:
        sys.path.append(dir_this)

    if not img_pathname.startswith("/"):
        dir_root = os.path.dirname(dir_this)
        img_pathname = "{}/{}".format(dir_root, img_pathname)

    id = ClfBeard.get_category_id(img_pathname)
    print("category id for {} is {}".format(img_pathname, id))
    return id

if __name__ == '__main__':
    filename = "_reserved_output_feature_gen/brd_image1_beard.jpg"
    do_exp(filename)
