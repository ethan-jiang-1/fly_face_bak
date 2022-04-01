
from bld_gen.model_inspect.bpy_data_inspect_base import BpyDataInsbase
from datetime import datetime
from bld_gen.utils_model.easy_dict import EasyDict
import importlib


class BpyDataMcdBurger(BpyDataInsbase):
    def __init__(self, bpy_data, renv=None):
        import bpy
        super(BpyDataMcdBurger, self).__init__(bpy_data, renv)
        self.scene = bpy.context.scene

    def inspect(self):
        self._inspect_top_collections()
        self._inspect_top_node_groups()
        self._inspect_all_materials()

    def _get_shot_info(self):
        now = datetime.now()
        shot_info = EasyDict()
        shot_info.gen_img_folder = self.get_gen_img_folder()
        shot_info.prefix = "{:04d}{:02d}{:02d}".format(now.year, now.month, now.day)
        shot_info.now = now
        shot_info.ndx = 0
        return shot_info

    def _find_all_shapekeys(self):
        import bld_gen.utils_model.finder_shapekeys_info as finder_shapekeys_info
        importlib.reload(finder_shapekeys_info)
        return finder_shapekeys_info.FinderShapekeysInfo.find_all_shapekeys_info(self.bpy_data)

    def _adjust_shapekeys(self, map_skm, shot_info=None):
        print(map_skm.keys())
        for key, lst_skm in map_skm.items():
            print("key", key)
            for val in [0.2, 0.4, 0.6, 0.8, 1.0, 0.0]:
                for skm in lst_skm:            
                    _, key, mesh, _ = skm
                    kb = mesh.shape_keys.key_blocks[key]
                    kb.value = val
                    if shot_info:
                        self.take_shot(shot_info)
                    else:
                        self.refresh_screen()
                    if val == 0.0:
                        self.refresh_screen()

    def adjust(self):
        map_skm = self._find_all_shapekeys()
        self._adjust_shapekeys(map_skm, shot_info=None)
        return True

    def adjust_capture_imgs(self):
        shot_info = self._get_shot_info()
        map_skm = self._find_all_shapekeys()
        self._adjust_shapekeys(map_skm, shot_info=shot_info)
        return True

def do_exp_show(renv):
    import bpy

    bpy_data = bpy.data 
    bd = BpyDataMcdBurger(bpy_data, renv=renv)
    bd.inspect()
    return bd.adjust()

def do_exp_capture(renv):
    import bpy

    bpy_data = bpy.data 
    bd = BpyDataMcdBurger(bpy_data, renv=renv)
    bd.inspect()
    return bd.adjust_capture_imgs()

def do_exp(renv):
    return do_exp_show(renv)


if __name__ == '__main__':
    do_exp(None)
