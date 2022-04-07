import os
import PySimpleGUI as sg

from collections import namedtuple
from pprint import pprint

from clf_net.clf_face import ClfFace
from clf_net.clf_bread import ClfBeard
from clf_net.clf_hair import ClfHair
from bld_gen.poster_query import PosterQuery

from utils.colorstr import log_colorstr

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
        log_colorstr("blue", "#SMP: process {} as {} in {}secs".format(img_filename, title, dt.total_seconds()))

        img_org, img_hair, img_face, img_beard = cls.get_bin_images(imgs, names)

        hair_id = ClfHair.get_category_id(img_hair)
        face_id = ClfFace.get_category_id(img_face)
        beard_id = ClfBeard.get_category_id(img_beard)
        log_colorstr("blue", "#SMP: hair_id:{}, face_id:{}, beard_id:{}".format(hair_id, face_id, beard_id))

        poster_pathname, _ = PosterQuery.get_poster(hair_id, beard_id, face_id)
        log_colorstr("blue", "#SMP: poster_pathname: {}".format(poster_pathname))
        
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
    from utils.plot_helper import PlotHelper

    img_filename = "utils_inspect/_sample_imgs/brd_image2.jpeg"

    smp_ret = SearchMatchedPoster.search_for_poster(img_filename)
    pprint(smp_ret)

    img_poster = None

    imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_face, smp_ret.img_beard, img_poster]
    names = ["org", "hair[{}]".format(smp_ret.hair_id), "face[{}]".format(smp_ret.face_id), "beard[{}]".format(smp_ret.beard_id), "poster"]

    PlotHelper.plot_imgs(imgs, names)
    
def start_work():
    from utils.plot_helper import PlotHelper
    
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('选择图片')],
            [sg.Button('OK'), sg.Button('Cancel')],
            [sg.Text('File', size=(5, 1)), sg.Input(size=(40, 1)), sg.FileBrowse(key='-File-')]
            ]

    window = sg.Window('人脸识别', layout)
    event, values = window.read()

    while True:
        event, values = window.read()
        print(f'Event: {event}')
        
        if event == 'OK':
            if (values[0] == ""):
                sg.popup("请选择照片");
            elif (not os.path.exists(values[0])):
                sg.popup("照片不存在，请重新选择")
            else:
                print(values[0])
                smp_ret = SearchMatchedPoster.search_for_poster(values[0])
                pprint(smp_ret)

                img_poster = None
                imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_face, smp_ret.img_beard, img_poster]
                names = ["org", "hair[{}]".format(smp_ret.hair_id), "face[{}]".format(smp_ret.face_id), "beard[{}]".format(smp_ret.beard_id), "poster"]
                PlotHelper.plot_imgs(imgs, names)
        elif event in (None, 'Cancel'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')

if __name__ == '__main__':
    # do_exp()
    start_work()
