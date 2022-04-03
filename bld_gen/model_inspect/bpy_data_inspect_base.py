import os
from bld_gen.utils_ui.bd_interaction import BdInteraction
from bld_gen.utils_ui.bd_interaction_shapekeys import BdInteractionShapeKeys

ENV_DISABLE_REFRESH = "DISABLE_REFRESH"
ENV_DISABLE_RENDER = "DISABLE_RENDER"

class BpyDataInsbase(object):
    def __init__(self, bpy_data, renv=None, inspect_flags="ALL"):
        self.bpy_data = bpy_data
        self.renv = renv
        self.inspect_flags = inspect_flags         

        self.cltname2objs = {}
        self.ngname2nodes = {}

        self.name2bones = {}
        self.name2materials = {}

        self.debug = True

    def inspect_all(self):
        self._inspect_top_collections()
        self._inspect_top_node_groups()
        self._inspect_top_armatures()

        self._inspect_all_objects()
        self._inspect_all_meshes()
        self._inspect_all_materials()

    def _inspect_top_collections(self):
        clts = self.bpy_data.collections 
        for idx, clt in enumerate(clts):
            self._inspect_collection(clt)

    def _inspect_collection(self, clt):
        cltname = clt.name
        if cltname not in self.cltname2objs:
            self.cltname2objs[cltname] = []
        for obj in clt.objects:
            self.cltname2objs[cltname].append(obj) 

    def _inspect_top_node_groups(self):
        ngs = self.bpy_data.node_groups
        for idx, ng in enumerate(ngs):
            if ng.name not in self.ngname2nodes:
                self.ngname2nodes[ng.name] = ng
            self._inspect_node_group(ng)

    def _inspect_node_group(self, ng):
        pass

    def _inspect_top_armatures(self):
        amts = self.bpy_data.armatures
        for idx, amt in enumerate(amts):
            self._inspect_armature(amt) 

    def _inspect_armature(self, amt):
        self._inspect_armature_bones(amt.bones)
        self._inspect_armature_layers(amt.layers)
        self._inspect_armature_layers_protected(amt.layers_protected)

    def _inspect_armature_bones(self, bones):
        for idx, bone in enumerate(bones):
            self.name2bones[bone.name] = bone

    def _inspect_armature_pose(self, pose):
        pass

    def _inspect_armature_layers(self, layers):
        pass

    def _inspect_armature_layers_protected(self, layers_protected):
        pass

    def _inspect_all_objects(self):
        objs = self.bpy_data.objects
        missed = []
        for idx, obj in enumerate(objs):
            if not self._inspect_object(obj):
                missed.append(obj.name)

        print("missed objs", missed)

    def _inspect_object(self, obj):
        return False 

    def _inspect_all_meshes(self):
        meshes = self.bpy_data.meshes
        for idx, mesh in enumerate(meshes):
            self._inspect_mesh(mesh)

    def _inspect_mesh(self, mesh):
        pass

    def _inspect_all_materials(self):
        materials = self.bpy_data.materials
        for idx, mat in enumerate(materials):
            if mat.name not in self.name2materials:
                self.name2materials[mat.name] = mat
                self._inspect_material(mat)

    def _inspect_material(self, mat):
        pass

    def _find_mesh_belong_to(self, mesh):
        for key, obj in self.all_objs.items():
            if obj.data == mesh:
                return key, obj
        return None, None

    def send_dev_msg(self, msg):
        BdInteraction.send_dev_msg(self.renv, msg)

    def _restore_org(self):
        BdInteraction.refresh_screen()

    def play_shapekeys_all(self, val_range=None):
        self._restore_org()

        if not hasattr(self, "shapekeys_info"):
            self.shapekeys_info = BdInteractionShapeKeys.get_shapekeys_info(self.bpy_data)
        BdInteractionShapeKeys.play_shapekeys_all(self.renv, self.shapekeys_info, val_range=val_range)

        self._restore_org()
        return True

    def play_shapekeys(self, val_range=None, sel_key_names=[]):
        self._restore_org()

        if not hasattr(self, "shapekeys_info"):
            self.shapekeys_info = BdInteractionShapeKeys.get_shapekeys_info(self.bpy_data)
        BdInteractionShapeKeys.play_shapekeys(self.renv, self.shapekeys_info, val_range=val_range, sel_key_names=sel_key_names)
   
        self._restore_org()
        return True

    def locate_object_at_collection(self, obj_name, collection_name):
        if not hasattr(self, "cltname2objs"):
            return None
        if not collection_name in self.cltname2objs.keys():
            return None
        objs = self.cltname2objs[collection_name]
        for obj in objs:
            if obj.name == obj_name:
                return obj
        return None

    def get_gen_img_folder(self):
        prj_pathname = self.bpy_data.filepath
        basename = os.path.basename(prj_pathname)
        if basename.endswith(".blend"):
            basename = basename.replace(".blend", "")
        gen_img_folder = "{}{}_gen_{}".format(self.renv.get_root_dir(), os.sep, basename)
        os.makedirs(gen_img_folder, exist_ok=True)
        return gen_img_folder

    def take_shot(self, shot_info):
        from bld_gen.utils_ui.bd_render import BdRender
        if shot_info is not None:
            if ENV_DISABLE_RENDER in os.environ.keys():
                if os.environ[ENV_DISABLE_RENDER].lower() == "true":
                    print("SKIP render - {}={}".format(ENV_DISABLE_RENDER,  os.environ[ENV_DISABLE_RENDER]))
                    return self.refresh_screen()
            img_pathname = "{}{}{}_{:06d}.png".format(shot_info.gen_img_folder, os.sep, shot_info.prefix, shot_info.ndx)
            BdRender.render_scene_to_img(img_pathname)  
            shot_info.ndx += 1
            return True
        else:
            print("SKIP render - not shot info")
            self.refresh_screen()  
            return False

    def refresh_screen(self):
        if ENV_DISABLE_REFRESH in os.environ.keys():
            if os.environ[ENV_DISABLE_REFRESH].lower() == "true":
                print("SKIP refresh - {}={}".format(ENV_DISABLE_REFRESH,  os.environ[ENV_DISABLE_REFRESH]))
                return False
        import bpy
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        return True
