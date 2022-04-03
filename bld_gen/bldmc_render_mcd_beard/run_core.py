
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

    def _alter_obj_hide(self, obj, hide):
        #import bpy
        if obj.hide_render != hide:
            obj.hide_render = hide
            #bpy.ops.object.hide_render_set(hide)
            self.refresh_screen()
        if obj.hide_viewport != hide:
            obj.hide_viewport = hide
            #bpy.ops.object.hide_view_set(hide) 
            self.refresh_screen()

    def _select_one_visible_obj_only(self, sel_obj, objs):
        for obj in objs:
            if obj is None:
                continue
            if obj == sel_obj:
                self._alter_obj_hide(obj, False)
            else:
                self._alter_obj_hide(obj, True)
        self.refresh_screen()

    def _deselect_all_obj(self, objs):
        for obj in objs:
            if obj is None:
                continue
            self._alter_obj_hide(obj, True)

    def _should_skip_shapekey(self, key, beard, hair):
        key_lower = key.lower()
        if key_lower == "basis":
            return True
        if hair is None:
            if key_lower.startswith("hair"):
                return True
        if beard is None:
            if key_lower.startswith("beard"):
                return True
        return False        

    def _adjust_shapekeys(self, map_skm, shot_info=None):
        beard_objs = self.cltname2objs["Collection beard"]
        hair_male_objs = self.cltname2objs["Collection hair_male"]
        hair_female_objs = self.cltname2objs["Collection hair_female"]

        self._deselect_all_obj(beard_objs)
        self._deselect_all_obj(hair_male_objs)
        self._deselect_all_obj(hair_female_objs)

        beard_objs.insert(0, None)
        hair_objs = []
        hair_objs.append(None)
        hair_objs.extend(hair_male_objs)
        hair_objs.extend(hair_female_objs)
        total_loop = len(beard_objs) * len(hair_objs)
        vals = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0]
        total_shot = total_loop * len(vals) * len(map_skm.keys())
        print("total loop: {}, total_shot?: {}", total_loop, total_shot)

        for bidx, beard in enumerate(beard_objs):
            for hidx, hair in enumerate(hair_objs):
                if hair in hair_female_objs:
                    if beard is not None:
                        continue

                loop_name = "{:03d}/{:03d}:b{}h{}".format(bidx*len(beard_objs) + hidx, total_loop, bidx, hidx)
                self._select_one_visible_obj_only(hair, hair_objs)
                self._select_one_visible_obj_only(beard, beard_objs)
                self.send_dev_msg(loop_name)   
                print()
                print(loop_name)
                print(map_skm.keys())
                for key, lst_skm in map_skm.items():
                    if self._should_skip_shapekey(key, beard, hair):
                        continue
                    print("play key", key)
                    self.send_dev_msg("{}:{}".format(loop_name, key))                    
                    for val in vals:
                        for skm in lst_skm:            
                            _, key, mesh, _ = skm
                            if self._should_skip_shapekey(key, beard, hair):
                                continue
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
