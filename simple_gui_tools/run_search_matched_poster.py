import os
import sys
import PySimpleGUI as sg
#import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk) # noqa: F401
from pprint import pprint


def _check_run_path():
    dir_root = os.path.dirname(os.path.dirname(__file__))
    if dir_root not in sys.path:
        sys.path.append(dir_root)

def main_page():
    version = 1.2
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
    from imgp_search_matched_poster import SearchMatchedPoster
    
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('功能说明：识别所选照片的发型/脸型/胡子，并随机匹配卡通形象', text_color="blue")],
              [sg.Button('OK'), sg.Button('Exit'), sg.Radio("男", group_id=1, key="radio_male", default=True), sg.Radio("女", group_id=1, key="radio_female")],
              [sg.Text('选择文件', auto_size_text=True), sg.Input(size=(40, 1)), sg.FileBrowse(key='-File-')],
             ]

    window = sg.Window('人脸识别(单个文件)', layout, resizable=True)
    
    while True:
        event, values = window.read()
        print(f'Event: {event}')
        
        if event == 'OK':
            if (values[0] == ""):
                sg.popup("请选择照片")
            elif (not os.path.exists(values[0])):
                sg.popup("照片不存在，请重新选择")
            else:
                male = values["radio_male"]
                gender = "M" if male else "F"
                print("->file: %s, gender: %s" % (os.path.basename(values[0]), gender))
                
                try:
                    smp_ret = SearchMatchedPoster.search_for_poster_with_gender(values[0], gender)
                    pprint(smp_ret)
                except Exception as ex:
                        print("<><run_search_matched_poster.dep_one_by_one>error: %s" % ex)
                        continue

                img_poster = cv2.imread(smp_ret.poster_pathname)
                imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_face, smp_ret.img_beard, img_poster]
                names = ["org", "hair[{}]".format(smp_ret.hair_id), "face[{}]".format(smp_ret.face_id), "beard[{}]".format(0 if smp_ret.beard_id == 0 else 1), "poster"]
                PlotHelper.plot_imgs(imgs, names)
        elif event in (None, 'Exit'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')
    
def dep_one_by_one():
    import cv2
    from utils.plot_helper import PlotHelper
    from imgp_search_matched_poster import SearchMatchedPoster
    
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('功能说明：识别所选发型子目录中所有照片的发型/脸型/胡子，并随机匹配卡通形象', text_color="blue")],
              [sg.Button('OK'), sg.Button('Exit'), sg.Radio("男", group_id=1, key="radio_male", default=True), sg.Radio("女", group_id=1, key="radio_female")],
              [sg.Text('选择目录', auto_size_text=True), sg.Input(size=(40, 1)), sg.FolderBrowse(key='-Folder-', initial_folder=os.path.dirname(__file__))],
              [sg.TabGroup(key="tabgroup", expand_x=True, expand_y=True, layout=[])]
             ]
    empty_layout = []

    window = sg.Window('人脸识别(指定目录)', layout, resizable=True)
    tabgroup = window.Element("tabgroup")

    while True:
        event, values = window.read()
        print(f'Event: {event}')
        
        if event == 'OK':
            if (values[0] == ""):
                sg.popup("请选择目录")
            elif (not os.path.exists(values[0])):
                sg.popup("目录不存在，请重新选择")
            else:
                import matplotlib.pyplot as plt
                plt.ioff()
                tabgroup.layout(empty_layout)
                
                male = values["radio_male"]
                gender = "M" if male else "F"
                
                file_list = sorted(os.listdir(values[0]))
                index = 0
                for file in file_list:
                    if (file.startswith(".")):
                        continue

                    print("->file: %s, gender: %s" % (os.path.basename(values[0]), gender))
                    try:
                        smp_ret = SearchMatchedPoster.search_for_poster_with_gender("{}/{}".format(values[0], file), gender)
                        pprint(smp_ret)
                    except Exception as ex:
                        print("<><run_search_matched_poster.dep_one_by_one>error: %s" % ex)
                        continue

                    img_poster = cv2.imread(smp_ret.poster_pathname)
                    imgs = [smp_ret.img_org, smp_ret.img_hair, smp_ret.img_face, smp_ret.img_beard, img_poster]
                    names = ["org", "hair[{}]".format(smp_ret.hair_id), "face[{}]".format(smp_ret.face_id), "beard[{}]".format(0 if smp_ret.beard_id == 0 else 1), "poster"]
                    fig = PlotHelper.plot_imgs(imgs, names, return_fig=True)
                    canvas = sg.Canvas(key="canvas{}".format(index), size=(1200, 600), expand_x=True, expand_y=True)
                    tabgroup.add_tab(sg.Tab(file, expand_x=True, expand_y=True, layout=[[get_result(smp_ret, img_poster, gender)], [canvas]]))
                    draw_figure(canvas.TKCanvas, fig)
                    index += 1
                
                tabgroup.set_size(size=(1200, 600))
                window.refresh()
        elif event in (None, 'Exit'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def get_result(smp_ret, image, gender):
    style_code = "H{:02d}B{:02d}F{:02d}{}".format(smp_ret.hair_id, smp_ret.beard_id, smp_ret.face_id, gender)
    if smp_ret.poster_pathname is not None and image is not None:
        return sg.Text("识别正确，类型码：{}".format(style_code), text_color="green", background_color="white")
    else:
        return sg.Text("识别错误，类型码：{}，找不到性别为[{}]头发[{}]胡子[{}]脸型[{}]的卡通形象".format(style_code, gender, smp_ret.hair_id, smp_ret.beard_id, smp_ret.face_id), text_color="#dd5145", background_color="white")
                    

if __name__ == '__main__':
    _check_run_path()
    main_page()
