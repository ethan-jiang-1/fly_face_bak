import os
import PySimpleGUI as sg

#from collections import namedtuple
from pprint import pprint

from impg_search_matched_poster import SearchMatchedPoster

def start_work():
    import cv2
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
                sg.popup("请选择照片")
            elif (not os.path.exists(values[0])):
                sg.popup("照片不存在，请重新选择")
            else:
                print(values[0])
                smp_ret = SearchMatchedPoster.search_for_poster(values[0])
                pprint(smp_ret)

                img_poster = cv2.imread(smp_ret.poster_pathname)
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
