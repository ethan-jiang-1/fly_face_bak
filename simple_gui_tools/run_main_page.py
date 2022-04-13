import os
import sys
import PySimpleGUI as sg

dir_root = os.path.dirname(os.path.dirname(__file__))
if dir_root not in sys.path:
    sys.path.append(dir_root)

from run_show_hairstyles import analysis_selected_folder as analysis_hair
from run_show_beardstyles import analysis_selected_folder as analysis_beard
from run_search_matched_poster import main_page as get_cartoon_character
from run_batch_analysis_face import start_work as batch_analysis_face
from run_resize_image import start_work as resize_image

def show_form():
    sg.theme('Light Blue 2')
    
    window_size = (420, 180)
    frame_size = (window_size[0] - 30, 45)
    
    frame_top = [sg.Button("退出", key="exit")]
    frame_edge = sg.Frame(layout=[[sg.Button("发型分析", key="hair_style"), sg.Button("胡子分析", key="beard_style")]], title="边界分析", size=frame_size)
    frame_composite = sg.Frame(layout=[[sg.Button("卡通形象", key="cartoon"), sg.Button("批量分析", key="batch_analysis")]], title="综合分析", size=frame_size)
    frame_other = sg.Frame(layout=[[sg.Button("图片大小", key="resize_image")]], title="其他", size=frame_size)
    layout = [
            #   [frame_top],
              [frame_edge],
              [frame_composite],
              [frame_other]
             ]
    
    window = sg.Window('工具集', layout, size=window_size)

    while True:
        event, values = window.read()
        print(f'Event: {event}, Values: {values}')
        
        if event == 'hair_style':
            analysis_hair()
        elif event == 'beard_style':
            analysis_beard()
        elif event == 'cartoon':
            get_cartoon_character()
        elif event == 'batch_analysis':
            batch_analysis_face()
        if event == 'resize_image':
            resize_image()
        elif event in (None, 'exit'):
            break
        
    window.close()

if __name__ == '__main__':
    show_form()