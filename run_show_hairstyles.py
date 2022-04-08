#!/usr/bin/env python

import os
import PySimpleGUI as sg

from imgp_show_hairstyles import do_extract_hairstyles


def analysis_selected_folder():
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('选择一个目录', size=(15, 1), auto_size_text=True, justification='left')], 
              [sg.InputText(size=(80, 1)), sg.FolderBrowse(initial_folder=os.path.dirname(__file__))], 
              [sg.Button('OK'), sg.Button('Exit')]]
    
    window = sg.Window('发型分析', layout)
    
    while True:
        event, values = window.read()
        print(f'Event: {event}, Values: {values}')
        
        if event == 'OK':
            if (values[0] == ''):
                sg.popup('请选择发型图片所在的目录！')
            elif not os.path.exists(values[0]):
                sg.popup('目录不存在，请重新选择！')
            else:
                src_dirs = []
                folder_list = sorted(os.listdir(values[0]))
                for folder in folder_list:
                    folder = "{}/{}".format(values[0], folder)
                    # if not os.path.isfile(folder) and not str(folder).startswith("."):
                    # if not os.path.isfile(folder):
                    if os.path.isdir(folder):
                        src_dirs.append(folder)
                
                if len(src_dirs) == 0:
                    src_dirs.append(values[0])
                
                dst_dir = "_reserved_output_hair_styles"
                do_extract_hairstyles(src_dirs, dst_dir)
        elif event in (None, 'Exit'):
            break
    window.close()

 
if __name__ == '__main__':
    analysis_selected_folder()
