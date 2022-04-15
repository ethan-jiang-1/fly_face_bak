import os
#import sys 
import random

class PosterQueryLocal(object):
    kls_output_folder = None
    kls_map_subfolders = None

    @classmethod
    def get_poster(cls,  hair_id, beard_id, face_id, gender):
        from utils.colorstr import log_colorstr
        output_folder = cls._get_output_folder()
        if output_folder is None:
            log_colorstr("red", "no local output folder found from bld_gen")
            return None, None

        # ethan hack: we only support one type of beard for now (2020/4/15)
        if beard_id > 0:
            beard_id = 1

        folder_id = "H{:02d}B{:02d}F{:02d}{}".format(hair_id, beard_id, face_id, gender)
        map_subfolder = cls._get_mapped_output_subfolders()
        if map_subfolder is None:
            log_colorstr("red", "no local generated posters found")
            return None, None 
        if folder_id not in map_subfolder:
            log_colorstr("red", "no local generated posters found match folder_id:{} ".format(folder_id))
            return None, None

        matched_folder = map_subfolder[folder_id]

        picked_pathname, _ = cls._random_pick_a_filename(matched_folder, folder_id)
        log_colorstr("green", "matched poster found at {} to match {}".format(picked_pathname, folder_id))

        return picked_pathname, None

    @classmethod
    def _random_pick_a_filename(cls, matched_folder, folder_id):
        names = os.listdir(matched_folder)
        len_names = len(names)
        for _ in range(5):
            rand_int = random.randint(0, len_names - 1)
            picked_name = names[rand_int]
            if picked_name.startswith(folder_id) and picked_name.endswith(".jpeg"):
                break
        picked_pathname = matched_folder + os.sep + picked_name
        return picked_pathname, None

    @classmethod
    def _get_mapped_output_subfolders(cls):
        if cls.kls_map_subfolders is not None:
            return cls.kls_map_subfolders

        cls.kls_map_subfolders = {}
        output_folder = cls._get_output_folder()
        if not os.path.isdir(output_folder):
            return None

        names = os.listdir(output_folder)
        for name in names:
            subfolder_name = output_folder + os.sep + name 
            if not os.path.isdir(subfolder_name):
                continue
            if not name.startswith("H"):
                continue

            strs = name.split("-")
            cls.kls_map_subfolders[strs[0]] = subfolder_name
        return cls.kls_map_subfolders

    @classmethod
    def _get_output_folder(cls):
        if cls.kls_output_folder is not None:
            return cls.kls_output_folder
        
        dir_root = os.path.dirname(os.path.dirname(__file__))
        output_folder = dir_root + os.sep + "_gen_render_m00"
        cls.kls_output_folder = output_folder
        return cls.kls_output_folder

def do_exp():
    hair_id = 0
    face_id = 0
    beard_id = 0

    output_folder = PosterQueryLocal._get_output_folder()
    print(output_folder)

    filename, _ = PosterQueryLocal.get_poster(hair_id, face_id, beard_id, "M")
    print("post for hid:{} fid: {} bid: {} is {}".format(hair_id, face_id, beard_id, filename))
    return filename

if __name__ == '__main__':
    import sys
    dir_root = os.path.dirname(os.path.dirname(__file__))
    if dir_root not in sys.path:
        sys.path.append(dir_root)
    do_exp()
