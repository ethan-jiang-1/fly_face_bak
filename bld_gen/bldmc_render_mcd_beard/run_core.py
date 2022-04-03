
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

    def _select_visible_obj_only(self, sel_obj, objs):
        for obj in objs:
            if obj is None:
                continue
            if obj == sel_obj:
                obj.hide_render = False
                obj.hide_viewport = False
            else:
                obj.hide_render = True
                obj.hide_viewport = True

    def _deselect_all_obj(self, objs):
        for obj in objs:
            if obj is None:
                continue
            obj.hide_render = True
            obj.hide_viewport = True        

    def _adjust_shapekeys(self, map_skm, shot_info=None):
        beard_objs = self.cltname2objs["Collection beard"]
        hair_objs = self.cltname2objs["Collection hair"]

        self._deselect_all_obj(beard_objs)
        self._deselect_all_obj(hair_objs)
        self.refresh_screen()

        beard_objs.insert(0, None)
        hair_objs.insert(0, None)

        for beard in beard_objs:
            for hair in hair_objs:
                self._select_visible_obj_only(hair, hair_objs)
                self._select_visible_obj_only(beard, beard_objs)
                self.refresh_screen()    

                print(map_skm.keys())
                for key, lst_skm in map_skm.items():
                    key_lower = key.lower()
                    if key_lower == "basis":
                        continue
                    if hair is None:
                        if key_lower.startswith("hair"):
                            continue
                    if beard is None:
                        if key_lower.startswith("beard"):
                            continue 
                    print("play key", key, key_lower)
                    self.send_dev_msg(key)                    
                    for val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0]:
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
    #return do_exp_capture(renv)

def do_exp_auto_render(renv):
    return do_exp_capture(renv)

if __name__ == '__main__':
    do_exp(None)
