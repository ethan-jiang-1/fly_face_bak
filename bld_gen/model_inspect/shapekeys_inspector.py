import os
from bld_gen.utils_ui.bd_interaction import BdInteraction 
from bld_gen.utils_ui.bd_interaction_shapekeys import BdInteractionShapeKeys
from bld_gen.utils_ui.bd_interaction_vrm import BdInteractionVrm

DEFAULT_BS_STRENGTH = 0.0
DEFAULT_PLAY_MODE = "NATIVE|VRM|ARKIT",

class ShapeKeysInspector(object):
    def __init__(self, renv):
        self.renv = renv

        self.bs_strength = DEFAULT_BS_STRENGTH
        if "BS_STRENGTH" in os.environ:
            val = os.environ["BS_STRENGTH"]
            try:
                val = float(val)
                self.bs_strength = val
            except:
                pass
        self.play_mode = DEFAULT_PLAY_MODE
        if "PLAY_MODE" in os.environ:
            self.play_mode = os.environ["PLAY_MODE"]

        self.map_mesh_matched = {}

        self.debug_mode = True 

    def _do_find_grouped_shapekeys_info_info(self):
        import bpy 
        bpy_data = bpy.data
        return BdInteractionShapeKeys.get_shapekeys_info(bpy_data, debug=self.debug_mode)

    def _do_find_vrm_info(self):
        import bpy 
        bpy_data = bpy.data
        return BdInteractionVrm.get_vrm_info(bpy_data, debug=self.debug_mode)

    def send_dev_msg(self, msg):
        BdInteraction.send_dev_msg(self.renv, msg)

    def _dump_bs_txt(self, shapekeys_info, vrm_info):
        import bpy

        if os.path.isfile(bpy.data.filepath):
            bsfilename_txt = bpy.data.filepath + ".txt"
        else:
            bsfilename_txt = "new_inspect.txt"

        txt = ""
        with open(bsfilename_txt, "wt+") as f:

            if hasattr(shapekeys_info, "get_dump_bs_txt"):
                txt += shapekeys_info.get_dump_bs_txt()

            if hasattr(vrm_info, "get_dump_bs_txt"):
                txt += vrm_info.get_dump_bs_txt()

            f.write(txt)
 
    def _adjust_by_shapekeys_info(self, shapekeys_info):
        BdInteractionShapeKeys.play_shapekeys_all(self.renv, shapekeys_info)
        BdInteractionShapeKeys.restore_shapekeys(self.renv, shapekeys_info)

    def _adjust_by_vrm_info(self, vrm_info):
        BdInteractionVrm.play_vrm_eye_movement(self.renv, vrm_info)
        BdInteractionVrm.play_vrm_blendshape_all(self.renv, vrm_info, self.bs_strength)
 
    def inspect(self):
        shapekeys_info = self._do_find_grouped_shapekeys_info_info()
        vrm_info = self._do_find_vrm_info()

        self._dump_bs_txt(shapekeys_info, vrm_info)

        if vrm_info.has_support_data():
            if "VRM" in self.play_mode:
                self.send_dev_msg("play VRM blendshape")
                self._adjust_by_vrm_info(vrm_info)
        
        if shapekeys_info.has_support_data():
            if "NATIVE" in self.play_mode:
                self.send_dev_msg("play NATIVE shape keys")
                self._adjust_by_shapekeys_info(shapekeys_info)


def do_inspect(renv):
    print("##inspect shapekeys")

    ski = ShapeKeysInspector(renv)
    return ski.inspect()
