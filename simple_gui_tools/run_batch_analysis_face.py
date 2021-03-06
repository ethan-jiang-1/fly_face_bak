import os
import sys
import PySimpleGUI as sg

from posixpath import basename
from collections import namedtuple

SMP_RESULT = namedtuple('SMP_RESULT', "img_org img_hair img_beard img_face hair_id beard_id face_id poster_pathname") 

def _check_run_path():
    dir_root = os.path.dirname(os.path.dirname(__file__))
    if dir_root not in sys.path:
        sys.path.append(dir_root)

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
    def get_bin_images(cls, imgs, etnames):
        img_org = None
        img_hair = None 
        img_face = None
        img_beard = None
        for idx, name in enumerate(etnames):
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
    def search_for_poster(cls, img_filename, chk_hair, chk_beard, chk_face, gender):
        from clf_net.clf_face import ClfFace
        from clf_net.clf_beard import ClfBeard
        from clf_net.clf_hair import ClfHair
        from bld_gen.poster_query_local import PosterQueryLocal
        from utils.colorstr import log_colorstr

        cls.init_searcher()

        ffg_ret = cls.ffg.process_image(img_filename)
        imgs, etnames = ffg_ret.imgs, ffg_ret.etnames
        log_colorstr("blue", "#SMP: process {}".format(img_filename))

        img_org, img_hair, img_face, img_beard = cls.get_bin_images(imgs, etnames)

        bname = os.path.basename(img_filename)
        if gender not in ["F", "M"]:
            raise ValueError("not able to indentify gender from filename: {}".format(bname))

        hair_id = 0 if not chk_hair else ClfHair.get_category_id(img_hair, gender=gender)
        beard_id = 0 if not chk_beard else ClfBeard.get_category_id(img_beard, gender=gender)
        face_id = 0 if not chk_face else ClfFace.get_category_id(img_face, gender=gender)
        log_colorstr("blue", "#SMP: hair_id:{},  beard_id:{}, face_id:{}".format(hair_id, beard_id, face_id))

        poster_pathname, _ = PosterQueryLocal.get_poster(hair_id, beard_id, face_id, gender)
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
    version = 1.0
    sg.theme('Light Blue 2')
    
    layout = [[sg.Text('???????????????????????????????????????(???Version 1.2)???????????????(??????/??????/??????)??????????????????', text_color="blue")],
              [sg.Button('OK'), 
               sg.Button('Exit'), 
               sg.Radio("??????", key = "radio_hair", group_id = 1, default=True), 
               sg.Radio("??????", key = "radio_beard", group_id = 1), 
               sg.Radio("??????", key = "radio_face", group_id = 1), 
               sg.Radio("???", group_id=2, key="radio_male", default=True), 
               sg.Radio("???", group_id=2, key="radio_female")],
              [sg.Text('????????????', auto_size_text=True), 
               sg.Input(size=(40, 1)), 
               sg.FolderBrowse(key='folder_browse', initial_folder=os.path.dirname(__file__))],
              [sg.Multiline('????????????', key = "result", size=(56, 20))],
             ]

    window = sg.Window('????????????????????????({})'.format(version), layout)
    event, values = window.read()

    while True:
        event, values = window.read()
        print(f'Event: {event}, Values: {values}')
        
        if event == 'OK':
            check_hair = values["radio_hair"]
            check_beard = values["radio_beard"]
            check_face = values["radio_face"]
            selected_folder = values["folder_browse"]
        
            if (selected_folder == ""):
                window["result"].update("???????????????")
            elif (not os.path.exists(selected_folder)):
                window["result"].update("?????????????????????????????????")
            else:
                window["result"].update("????????????...")
                male = values["radio_male"]
                gender = "M" if male else "F"
                
                sub_folder_list = []
                sub_folders = sorted(os.listdir(selected_folder))
                for sub_folder in sub_folders:
                    if is_valid_folder(selected_folder, sub_folder):
                        sub_folder_list.append(sub_folder)
                
                if len(sub_folder_list) == 0:
                    window["result"].update("?????????????????????????????????Version 1.2")
                    continue
                
                result_list_all = []
                result_list_err = []
                for sub_folder in sub_folder_list:
                    if not is_valid_folder(selected_folder, sub_folder):
                        continue
                
                    sub_folder_full_path = "{}/{}".format(selected_folder, sub_folder)
                    if os.path.isdir(sub_folder_full_path):
                        file_list = sorted(os.listdir(sub_folder_full_path))
                        for file in file_list:
                            if not is_valid_file(sub_folder_full_path, file):
                                continue
                            
                            file_name = "{}/{}".format(sub_folder_full_path, file)
                            smp_ret = SearchMatchedPoster.search_for_poster(file_name, check_hair, check_beard, check_face, gender)
                            # pprint(smp_ret)
                            result_list_all.append("file: {}, sub_folder: {}, hair: {}, face: {}, beard: {}".format(file, sub_folder, smp_ret.hair_id, smp_ret.face_id, smp_ret.beard_id))
                            
                            hair_id_format = "{:02d}".format(smp_ret.hair_id)
                            beard_id_format = "{:02d}".format(smp_ret.beard_id)
                            face_id_format = "{:02d}".format(smp_ret.face_id)
                            
                            hair_style_error = (check_hair == True and hair_id_format != sub_folder)  # noqa:E712
                            beard_style_error = (check_beard == True and beard_id_format != sub_folder)  # noqa:E712
                            face_style_error = (check_face == True and face_id_format != sub_folder)  # noqa:E712
                            
                            if hair_style_error:
                                result_list_err.append("file: {}, sub_folder: {}, hair: {}".format(file, sub_folder, hair_id_format))
                            elif beard_style_error:
                                result_list_err.append("file: {}, sub_folder: {}, beard: {}".format(file, sub_folder, beard_id_format))
                            elif face_style_error:
                                result_list_err.append("file: {}, sub_folder: {}, face: {}".format(file, sub_folder, face_id_format))

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
                    window["result"].update("???????????????????????????????????????")
        elif event in (None, 'Exit'):
            break
        
    window.close()

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
    _check_run_path()
    start_work()
