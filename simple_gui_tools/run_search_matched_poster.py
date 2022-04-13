import os
import sys
import PySimpleGUI as sg
sys.path.append(os.getcwd())
from pprint import pprint
from impg_search_matched_poster import SearchMatchedPoster

def main_page():
    version = 1.0
    sg.theme('Light Blue 2')
    
    layout = [[sg.Button('选择文件', size=(9, 1)), sg.Button('选择目录', size=(9, 1)), sg.Button('退出', size=(9, 1))]]

    window = sg.Window('人脸识别工具({})'.format(version), layout, size=(240, 80))
    event, values = window.read()

    while True:
        event, values = window.read()
        print(f'Event: {event}')
        
        if event == '选择文件':
            dep_one_file()
        elif event == '选择目录':
            dep_one_by_one()
        elif event in (None, '退出'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')

def dep_one_file():
    import cv2
    from utils.plot_helper import PlotHelper
    
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('功能说明：识别所选照片的发型/脸型/胡子，并随机匹配卡通形象', text_color="blue")],
              [sg.Button('OK'), sg.Button('Exit')],
              [sg.Text('选择文件', auto_size_text=True), sg.Input(size=(40, 1)), sg.FileBrowse(key='-File-')]
             ]

    window = sg.Window('人脸识别(单个文件)', layout)
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
        elif event in (None, 'Exit'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')
    
def dep_one_by_one():
    import cv2
    from utils.plot_helper import PlotHelper
    
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('功能说明：识别所选发型子目录中所有照片的发型/脸型/胡子，并随机匹配卡通形象', text_color="blue")],
              [sg.Button('OK'), sg.Button('Exit')],
              [sg.Text('选择目录', auto_size_text=True), sg.Input(size=(40, 1)), sg.FolderBrowse(key='-Folder-', initial_folder=os.path.dirname(__file__))]
             ]

    window = sg.Window('人脸识别(指定目录)', layout)
    event, values = window.read()

    while True:
        event, values = window.read()
        print(f'Event: {event}')
        
        if event == 'OK':
            if (values[0] == ""):
                sg.popup("请选择目录")
            elif (not os.path.exists(values[0])):
                sg.popup("目录不存在，请重新选择")
            else:
                print(values[0])
                file_list = sorted(os.listdir(values[0]))
                for file in file_list:
                    if (file.startswith(".")):
                        continue
                    
                    smp_ret = SearchMatchedPoster.search_for_poster("{}/{}".format(values[0], file))
                    pprint(smp_ret)

                    img_poster = cv2.imread(smp_ret.poster_pathname)
                    imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_face, smp_ret.img_beard, img_poster]
                    names = ["org[{}]".format(file), "hair[{}]".format(smp_ret.hair_id), "face[{}]".format(smp_ret.face_id), "beard[{}]".format(smp_ret.beard_id), "poster"]
                    PlotHelper.plot_imgs(imgs, names)
        elif event in (None, 'Exit'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')

if __name__ == '__main__':
    main_page()