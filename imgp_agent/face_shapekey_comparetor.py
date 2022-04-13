
import os
import sys
from datetime import datetime

dir_this = os.path.dirname(__file__)
if dir_this not in sys.path:
    sys.path.append(dir_this)

from face_feature_generator import FaceFeatureGenerator
from imgp_common import FileHelper, PlotHelper


class FaceShapekeyComparetor(object):
    def __init__(self, debug=False):
        self.debug = debug
        self.ffg = FaceFeatureGenerator()

    def __del__(self):
        del self.ffg 

    def process(self, filenames):
        from utils.colorstr import log_colorstr

        sel_imgs = []
        sel_names = []
        for filename in filenames:
            if not os.path.isfile(filename):
                return None 

            d0 = datetime.now()
            imgs, etnames = self.ffg.process_image(filename)
            dt = datetime.now() - d0
            log_colorstr("blue", "total inference time: for {}: {:.3f}".format(os.path.basename(filename), dt.total_seconds()))

            for idx, name in enumerate(etnames):
                if name in ["org", "facemesh", "facepaint", "outline", "beard", "hair"]:
                    sel_names.append(name)
                    sel_imgs.append(imgs[idx])
    
        PlotHelper.plot_imgs_grid(sel_imgs, sel_names, mod_num=6)


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
