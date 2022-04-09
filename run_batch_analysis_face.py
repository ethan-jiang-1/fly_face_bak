import os
from statistics import multimode
import PySimpleGUI as sg

from collections import namedtuple
from pprint import pprint

from clf_net.clf_face import ClfFace
from clf_net.clf_bread import ClfBeard
from clf_net.clf_hair import ClfHair
from bld_gen.poster_query_local import PosterQueryLocal

from utils.colorstr import log_colorstr

SMP_RESULT = namedtuple('SMP_RESULT', "img_org img_hair img_beard img_face hair_id beard_id face_id poster_pathname") 

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
        beard_id = ClfBeard.get_category_id(img_beard)
        face_id = ClfFace.get_category_id(img_face)
        log_colorstr("blue", "#SMP: hair_id:{},  beard_id:{}, face_id:{}".format(hair_id, beard_id, face_id))

        poster_pathname, _ = PosterQueryLocal.get_poster(hair_id, beard_id, face_id)
        log_colorstr("blue", "#SMP: poster_pathname: {}".format(poster_pathname))
        
        smp_ret = SMP_RESULT(img_org=img_org,
                             img_hair=img_hair,
                             img_beard=img_beard,
                             img_face=img_face,
                             hair_id=hair_id,
                             beard_id=beard_id,
                             face_id=face_id,
                             poster_pathname=poster_pathname)
        return smp_ret

def start_work():
    import cv2
    from utils.plot_helper import PlotHelper
    
    version = 1.0
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('功能说明：选择发型的根目录(如Version 1.2)，检查所有发型分类是否正确', text_color="blue")],
              [sg.Button('OK'), sg.Button('Exit')],
              [sg.Text('选择目录', auto_size_text=True), sg.Input(size=(40, 1)), sg.FolderBrowse(key='-Folder-', initial_folder="/Users/gaobo/Workspace/Python/fly_face/_dataset_org_hair_styles/Version 1.1")],
              [sg.Multiline('输出结果', key = "result", size=(56, 20))],
             ]

    window = sg.Window('发型分类检查工具({})'.format(version), layout)
    event, values = window.read()

    while True:
        event, values = window.read()
        print(f'Event: {event}')
        
        if event == 'OK':
            if (values[0] == ""):
                window["result"].update("请选择目录")
            elif (not os.path.exists(values[0])):
                window["result"].update("目录不存在，请重新选择")
            else:
                window["result"].update("开始分析...")
                sub_folder_list = []
                sub_folders = sorted(os.listdir(values[0]))
                for sub_folder in sub_folders:
                    if is_valid_folder(values[0], sub_folder):
                        sub_folder_list.append(sub_folder)
                
                if len(sub_folder_list) == 0:
                    window["result"].update("请选择发型的根目录，如Version 1.2")
                    continue
                
                result_list_all = []
                result_list_err = []
                for sub_folder in sub_folder_list:
                    if not is_valid_folder(values[0], sub_folder):
                        continue
                
                    sub_folder_full_path = "{}/{}".format(values[0], sub_folder)
                    if os.path.isdir(sub_folder_full_path):
                        file_list = sorted(os.listdir(sub_folder_full_path))
                        for file in file_list:
                            if not is_valid_file(sub_folder_full_path, file):
                                continue
                            
                            file_name = "{}/{}".format(sub_folder_full_path, file)
                            smp_ret = SearchMatchedPoster.search_for_poster(file_name)
                            # pprint(smp_ret)
                            result_list_all.append("file: {}, sub_folder: {}, hair: {}, face: {}, beard: {}".format(file, sub_folder, smp_ret.hair_id, smp_ret.face_id, smp_ret.beard_id));
                            
                            hair_id_format = "{:02d}".format(smp_ret.hair_id)
                            if (hair_id_format != sub_folder):
                                result_list_err.append("file: {}, sub_folder: {}, hair: {:02d}, face: {:02d}, beard: {:02d}".format(file, sub_folder, smp_ret.hair_id, smp_ret.face_id, smp_ret.beard_id));

                print("----all list----")
                for index, line in enumerate(result_list_all):
                    print("->all: %s, %s" % (index, line))
                    
                print("----err list----")
                all_line = ""
                if len(result_list_err) > 0:
                    for index, line in enumerate(result_list_err):
                        print("->err: %s, %s" % (index, line))
                        all_line += line + "\n"
                    all_line += "total count: {}".format(len(result_list_err))
                    window["result"].update(all_line)
                else:
                    window["result"].update("扫描完毕，未发现类型不匹配")
        elif event in (None, 'Exit'):
            break
        print(str(values))
        
    window.close()
    print(f'You clicked {event}')

def is_valid_folder(folder_path, folder_name):
    folder_name = str(folder_name)
    full_name = "{}/{}".format(folder_path, folder_name)
    if not folder_name.startswith(".") and os.path.isdir(full_name):
        return True
    else:
        return False
    
def is_valid_file(file_path, file_name):
    file_name = str(file_name)
    full_name = "{}/{}".format(file_path, file_name)
    if not file_name.startswith(".") and os.path.isfile(full_name):
        return True
    else:
        return False

if __name__ == '__main__':
    start_work()
