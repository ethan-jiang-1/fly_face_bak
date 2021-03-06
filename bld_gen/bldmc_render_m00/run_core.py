
import importlib
from datetime import datetime

import os
import json
from collections import namedtuple
from pprint import pprint

from bld_gen.model_inspect.bpy_data_inspect_base import BpyDataInsbase
from bld_gen.utils_model.easy_dict import EasyDict
from bld_gen.utils_ui.colorstr import log_colorstr
from bld_gen.auto_render_cfg import AutoRenderCfg, AutoRenderEnv

CIT = namedtuple("CIT", "hidx hair bidx beard fidx face cmb_idx gender")
AR_MODEL_ID = "00"

class CombinationWalker(object):
    def __init__(self,  cltname2objs):
        hair_male_objs = cltname2objs["Collection hair_male"]
        hair_female_objs = cltname2objs["Collection hair_female"]
        hair_objs = []
        hair_objs.append(None)
        hair_objs.extend(hair_male_objs)
        hair_objs.extend(hair_female_objs)

        beard_objs = cltname2objs["Collection beard"]
        beard_objs.insert(0, None)

        face_objs = []
        face_objs.insert(0, None)

        self.hair_objs = hair_objs
        self.beard_objs = beard_objs
        self.face_objs = face_objs
        self.hair_male_objs = hair_female_objs 
        self.hair_female_objs = hair_female_objs

    def build_iterator(self):
        cits = []
        cmb_idx = -1
        for hidx, hair in enumerate(self.hair_objs):
            for bidx, beard in enumerate(self.beard_objs):
                for fidx, face in enumerate(self.face_objs):
                    if hair in self.hair_female_objs:
                        if beard is not None:
                            continue
                        gender = "F"
                    else:
                        gender = "M"
                    cmb_idx += 1

                    cit = CIT(hidx=hidx, 
                              hair=hair, 
                              bidx=bidx, 
                              beard=beard,
                              fidx=fidx,
                              face=face,
                              cmb_idx=cmb_idx, 
                              gender=gender)
                    cits.append(cit)
        return cits

    def get_combination_name(self, hidx, bidx, fidx, gender, ar_model_id, cmb_idx):
        hnum = 0
        hair_obj = self.hair_objs[hidx]
        if hair_obj is not None:
            hnum = int(hair_obj.name.split(".")[1])
        
        bnum = 0
        beard_obj = self.beard_objs[bidx]
        if beard_obj is not None:
            bnum = int(beard_obj.name.split(".")[1])
        
        fnum = 0
        face_obj = self.face_objs[fidx]
        if face_obj is not None:
            fnum = int(face_obj.name.split(".")[1])
        
        combination_name = "H{:02}B{:02}F{:02}{}-m{}c{:03}".format(hnum, bnum, fnum, gender, ar_model_id, cmb_idx)
        return combination_name


class BpyDataMcBeard(BpyDataInsbase):
    def __init__(self, bpy_data, renv=None, auto_render_filename=None):
        import bpy
        super(BpyDataMcBeard, self).__init__(bpy_data, renv)
        self.scene = bpy.context.scene

        self.ar_model_id = AR_MODEL_ID

        self.render_root = self.get_gen_img_folder()

        self.auto_render_filename = auto_render_filename
        self.auto_render_data = None

        self.env_max_variation = 6
        self.env_combination_id = -1
        self.env_render_engine = "EEVEE"

        if auto_render_filename is not None:
            self._prepare_auto_render()
        self._prepare_env()

    def _prepare_auto_render(self):
        self.auto_render_data = self._load_json_data(self.auto_render_filename)
        if self.auto_render_data is not None:
            self.render_root = "{}{}{}".format(self.renv.get_root_dir(), os.sep, self.auto_render_data["output_folder"])
        print()
        pprint("auto_render_data", self.auto_render_data)
        print()

    def _prepare_env(self):
        if AutoRenderEnv.ENV_AR_MAX_VARIATION in os.environ:
            self.env_max_variation = int(os.environ[AutoRenderEnv.ENV_AR_MAX_VARIATION])
        if AutoRenderEnv.ENV_AR_COMBINATION_ID in os.environ:
            self.env_combination_id = int(os.environ[AutoRenderEnv.ENV_AR_COMBINATION_ID])
        if AutoRenderEnv.ENV_AR_RENDER_ENGINE in os.environ:
            self.env_render_engine = os.environ[AutoRenderEnv.ENV_AR_RENDER_ENGINE]
        print()
        print("env_max_variation ", self.env_max_variation)
        print("env_combination_id", self.env_combination_id)
        print("env_render_engine ", self.env_render_engine)
        print()

    def _load_json_data(self, json_filename):
        dir_this = os.path.dirname(__file__)
        json_path = "{}{}{}".format(dir_this, os.sep, json_filename)

        json_data = AutoRenderCfg.get_config_data(json_path)
        if json_data is None:
            return None
        log_colorstr("yellow", "auto_render cfg parameters found in {}".format(json_path))
        return json_data

    def inspect(self):
        self._inspect_top_collections()
        self._inspect_top_node_groups()
        self._inspect_all_materials()

    def _get_shot_info_inc(self):
        now = datetime.now()
        shot_info = EasyDict()
        shot_info.mode = "inc"
        shot_info.gen_img_folder = self.get_gen_img_folder()
        shot_info.prefix = "{:04d}{:02d}{:02d}".format(now.year, now.month, now.day)
        shot_info.now = now
        shot_info.ndx = 0
        return shot_info

    def _get_shot_info_auto(self):
        now = datetime.now()
        shot_info = EasyDict()
        shot_info.mode = "auto"
        shot_info.gen_img_folder = self.render_root
        shot_info.prefix = None
        shot_info.now = now
        shot_info.ndx = 0
        shot_info.gen_img_name = None
        return shot_info        

    def _find_all_shapekeys(self):
        import bld_gen.utils_model.finder_shapekeys_info as finder_shapekeys_info
        importlib.reload(finder_shapekeys_info)
        return finder_shapekeys_info.FinderShapekeysInfo.find_all_shapekeys_info(self.bpy_data)

    def _alter_obj_hide(self, obj, hide):
        change = False
        if obj.hide_render != hide:
            change = True
        if obj.hide_viewport != hide:
            change = True
        if change:
            obj.hide_viewport = hide
            obj.hide_render = hide
            obj.hide_set(hide)
            self.refresh_screen()

    def _select_one_visible_obj_only(self, sel_obj, objs):
        for obj in objs:
            if obj is None:
                continue
            if obj == sel_obj:
                self._alter_obj_hide(obj, False)
            else:
                self._alter_obj_hide(obj, True)

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

    def _adjust_varity_in_combination(self, cw, map_skm, cmb_idx, hidx, hair_objs, bidx, beard_objs, fidx, face_objs, gender, shot_info):
        vals = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]

        hair = hair_objs[hidx]
        beard = beard_objs[bidx]

        self._select_one_visible_obj_only(hair, hair_objs)
        self._select_one_visible_obj_only(beard, beard_objs)

        combination_name = cw.get_combination_name(hidx, bidx, fidx, gender, self.ar_model_id, cmb_idx)
        log_colorstr("yellow", "new combination: {} ".format(combination_name))

        vidx = 0 
        for key, lst_skm in map_skm.items():
            if self._should_skip_shapekey(key, beard, hair):
                continue
            if self.debug:
                log_colorstr("yellow", "play key {} for {} in {} ".format(key, cmb_idx, combination_name))
            self.send_dev_msg("{}:{}".format(combination_name, key))                    
            for val in vals:
                for skm in lst_skm:            
                    _, key, mesh, _ = skm
                    if self._should_skip_shapekey(key, beard, hair):
                        continue

                    vidx += 1
                    kb = mesh.shape_keys.key_blocks[key]
                    kb.value = val
                
                    if shot_info:
                        if shot_info.mode == "auto":
                            shot_info.gen_img_folder = self.render_root + os.sep + combination_name
                            shot_info.gen_img_name = combination_name + "_{:06}.jpeg".format(vidx)
                        self.take_shot(shot_info)
                    self.refresh_screen()

                    if self.env_max_variation > 1 and vidx >= self.env_max_variation:
                        log_colorstr("blue", "stop play varitry as {} >= {}".format(vidx, self.env_max_variation))
                        return combination_name, vidx 
        return combination_name, vidx

    def _save_shot_info(self, combination_name, vcnt, dt_seconds, shot_info):
        if shot_info is None or shot_info.mode != "auto":
            return None

        gi = self._get_gen_info(vcnt, dt_seconds)

        filename = self.render_root + os.sep + combination_name + os.sep + combination_name + ".json"
        with open(filename, "wt+") as f0:
            json.dump(gi, f0, indent=4)            

    def _adjust_shapekeys(self, map_skm, shot_info=None):
        cw = CombinationWalker(self.cltname2objs)

        beard_objs = cw.beard_objs
        hair_objs = cw.hair_objs
        face_objs = cw.face_objs

        self._deselect_all_obj(beard_objs)
        self._deselect_all_obj(hair_objs)
        self._deselect_all_obj(face_objs)

        cvcnt = 0
        cmb_idx = 0
        cits = cw.build_iterator()
        for cit in cits:
            cmb_idx = cit.cmb_idx
            if self.env_combination_id != -1:
                if self.env_combination_id != cmb_idx:
                    continue 
    
            hidx, bidx, fidx = cit.hidx, cit.bidx, cit.fidx
            gender = cit.gender
            d0 = datetime.now()
            combination_name, vcnt = self._adjust_varity_in_combination(cw, map_skm, cmb_idx, hidx, hair_objs, bidx, beard_objs, fidx, face_objs, gender, shot_info)
            dt = datetime.now() - d0
            self._save_shot_info(combination_name, vcnt, dt.total_seconds(), shot_info)
            self._save_publishing_info(combination_name, vcnt, dt.total_seconds())
            if self.debug:
                log_colorstr("blue", "combination {} has {} vairation".format(combination_name, vcnt))
            cvcnt += vcnt

        if self.debug:
            log_colorstr("blue", "total combination: {} yield {} variation".format(cmb_idx, cvcnt))
        return cmb_idx, cvcnt

    def adjust(self):
        map_skm = self._find_all_shapekeys()
        self._adjust_shapekeys(map_skm, shot_info=None)
        return True

    def adjust_capture_imgs(self):
        shot_info = self._get_shot_info_inc()
        map_skm = self._find_all_shapekeys()
        self._adjust_shapekeys(map_skm, shot_info=shot_info)
        return True

    def _gen_render_imgs(self, map_skm, shot_info=None):
        self._adjust_shapekeys(map_skm, shot_info=shot_info)
        return True

    def _get_gen_info(self, vcnt, dt_seconds):
        from utils.git_version import git_versions_from_vcs
        from utils_ui.bd_render import BdRender
        import socket 

        gi = {}
        dir_root = self.renv.get_root_dir()
        git_info = git_versions_from_vcs(dir_root)

        gi = {"git": git_info, 
                "date": str(datetime.now()),
                "user": str(os.getlogin()),
                "host": str(socket.gethostname()),
                "vcnt": vcnt,
                "dt_secs_in_total": dt_seconds,
                "dt_secs_per_image": dt_seconds/vcnt,
                "ar_model_id": self.ar_model_id,
                "ar_render_engine": BdRender.get_engine_type()}
        return gi

    def _save_publishing_info(self, combination_name, vcnt, dt_seconds):
        filename = self.render_root  + os.sep + "publishing_info_m{}.json".format(self.ar_model_id)

        pi = {}
        if os.path.isfile(filename):
            with open(filename, "rt") as fr:
                pi = json.load(fr)

        gi = self._get_gen_info(vcnt, dt_seconds)
        pi[combination_name] = gi

        with open(filename, "wt+") as fw:
            json.dump(pi, fw, indent=4)

    def auto_render(self):
        from bld_gen.utils_ui.bd_render import BdRender
        if self.env_render_engine == "EEVEE":
            BdRender.setup_engine_eevee()
        elif self.env_render_engine == "CYCLES":
            BdRender.setup_engine_cycles()
        shot_info = self._get_shot_info_auto()
        map_skm = self._find_all_shapekeys()
        self._gen_render_imgs(map_skm, shot_info=shot_info)

        return True

def do_exp_show(renv):
    import bpy

    bpy_data = bpy.data 
    bd = BpyDataMcBeard(bpy_data, renv=renv)
    bd.inspect()
    return bd.adjust()

def do_exp_capture(renv):
    import bpy

    bpy_data = bpy.data 
    bd = BpyDataMcBeard(bpy_data, renv=renv)
    bd.inspect()
    return bd.adjust_capture_imgs()

def do_exp(renv):
    return do_exp_show(renv)
    #return do_exp_capture(renv)

def _load_json_data(json_filename):
    import os
    import json

    dir_this = os.path.dirname(__file__)
    json_path = "{}{}{}".format(dir_this, os.sep, json_filename)
    if not os.path.isfile(json_path):
        msg = "failed to find {}".format(json_path)
        log_colorstr("red", msg)
        raise ValueError(msg)
    with open(json_path, "rb") as f:
        json_data = json.load(f)
    log_colorstr("yellow", "auto_render parameters found in {}".format(json_path))
    return json_data

def do_exp_auto_render(renv):
    import bpy

    auto_render_filename = "auto_render.json"
    if "AUTO_RENDER" in os.environ:
        auto_render_filename = os.environ["AUTO_RENDER"]

    bpy_data = bpy.data 
    bd = BpyDataMcBeard(bpy_data, renv=renv, auto_render_filename=auto_render_filename)
    bd.inspect()
    rc = bd.auto_render()
    return rc

if __name__ == '__main__':
    do_exp(None)
