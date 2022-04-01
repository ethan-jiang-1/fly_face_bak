from bld_utils_model.easy_dict import EasyDict
from collections import OrderedDict
import pprint

class VrmInfo(EasyDict):
    def __init__(self, *args, **kwargs):
        super(VrmInfo, self).__init__(*args, **kwargs)

        self.map_sorted_b2s = OrderedDict()
        self.bone_names_in_vrm0 = []
        self.eye_left_value_in_vrm0 = None
        self.eye_right_value_in_vrm0 = None 
        self.eye_left_bone_in_vrm0 = None
        self.eye_right_bone_in_vrm0 = None 
        self.eye_left_pose_bone_in_vrm0 = None
        self.eye_right_pose_bone_in_vrm0 = None 

    def get_dump_bs_txt(self):
        if not self.has_support_data():
            return ""

        map_sorted_b2s = self.map_sorted_b2s
        txt = "\nVRM_INFO:\n"
        txt += "#blendshape(vrm) to shapekey(native):\n"
        for key, val in map_sorted_b2s.items():
            txt += "  blendshape: {} => shapekey(s) ".format(key)
            txt += pprint.pformat(val, indent=4)
            txt += "\n"
        txt += "\n"
        if self.bone_names_in_vrm0 and len(self.bone_names_in_vrm0) > 0:
            txt += "  bone_names_in_vrm0: " + str(self.bone_names_in_vrm0) + "\n"
        if self.eye_left_value_in_vrm0 is not None:
            txt += "  eye_left: \t{}\n".format(self.eye_left_value_in_vrm0)
        if self.eys_right_value_in_vrm0 is not None:
            txt += "  eye_right:\t{}\n".format(self.eye_right_value_in_vrm0)
        return txt

    def has_support_data(self):
        map_sorted_b2s = self.map_sorted_b2s
        if len(map_sorted_b2s.keys()) == 0:
            return False
        return True        

    def is_valid(self):
        if not self.has_support_data():
            return False
        return True

class FinderVrmInfo(object):
    @classmethod
    def find_info(cls, bpy_data):

        vrm_info = VrmInfo()
        cls._fill_vrm__blendshape(bpy_data, vrm_info)

        if not vrm_info.is_valid():
            print("ERROR: vrm_info is not valid")

        return vrm_info

    @classmethod
    def _get_armature(cls, bpy_data):
        armature = bpy_data.objects.get("Armature")
        if armature is None:
            return None
        if armature.type != "ARMATURE":
            return None
        if armature.data is None:
            return None
        return armature

    @classmethod
    def _fill_vrm__blendshape(cls, bpy_data, vrm_info):
        armature = cls._get_armature(bpy_data)
        if armature is None:
            return False

        if not hasattr(armature.data, "vrm_addon_extension"):
            return False    
        
        vae = armature.data.vrm_addon_extension
        if hasattr(vae, "vrm1"):
            cls._check_vrm1_data(bpy_data, vae, vrm_info)
        if hasattr(vae, "vrm0"):
            cls._check_vrm0_data(bpy_data, vae, vrm_info) 
        return True

    @classmethod
    def _check_vrm1_data(cls, bpy_data, vae, vrm_info):
        return cls._check_vrm1_Vrm1HumanBonePropertyGroup(bpy_data, vae, vrm_info)

    @classmethod
    def _check_vrm1_Vrm1HumanBonePropertyGroup(cls, bpy_data, vae, vrm_info):
        vrm1  = vae.vrm1
        if not hasattr(vrm1, "vrm"):
            return False
        vrm = vrm1.vrm
        if "humanoid" not in vrm:
            return False 

        humanoid = vrm["humanoid"]
        right_eye, left_eye = None, None 
        obj_right_eye, obj_left_eye = None, None
        human_bones = humanoid["human_bones"]
        bone_names = list(human_bones.keys())
        print("human_bones in vrm1", bone_names, len(bone_names))
        if "right_eye" in human_bones:
            right_eye = human_bones["right_eye"]
            obj_right_eye = right_eye["node"]["link_to_bone"]
        if "left_eye" in human_bones:
            left_eye = human_bones["left_eye"]
            obj_left_eye = left_eye["node"]["link_to_bone"]
        print(obj_right_eye)
        print(obj_left_eye)
        vrm_info.bone_names_in_vrm1 = bone_names
        return True

    @classmethod
    def _check_vrm0_data(cls, bpy_data, vae, vrm_info):
        cls._check_vrm0_Vrm0BlendShapeGroupPropertyGroup(bpy_data, vae, vrm_info)
        cls._check_vrm0_Vrm0HumanoidPropertyGroup(bpy_data, vae, vrm_info)

    @classmethod
    def _check_vrm0_Vrm0BlendShapeGroupPropertyGroup(cls, bpy_data, vae, vrm_info):
        #blend_shape_master.blend_shape_groups
        vrm0 = vae.vrm0
        bsm = vrm0.blend_shape_master
        bsgs = bsm.blend_shape_groups

        map_sorted_b2s = OrderedDict()
        map_mesh_matched = {}
        for idx, bsg in enumerate(bsgs):
            print(idx, bsg)
            for im, mv in enumerate(bsg.material_values):
                print("", im, mv)
            lst_iwo = []
            for ib, bind in enumerate(bsg.binds):
                index_sk = bind.index
                weight_b2s = bind.weight
                mesh = bind.mesh
                mlm = mesh.link_to_mesh
                obj_parent = mlm.parent
                mesh_found = cls._get_mesh_obj(map_mesh_matched, obj_parent, mesh.value)
                shape_key = cls._get_shape_key(index_sk, mesh_found)
                print("", ib, index_sk, weight_b2s, mesh.value, shape_key)
                lst_iwo.append((index_sk, weight_b2s, mesh.value, shape_key))
            map_sorted_b2s[bsg.name] =lst_iwo
        vrm_info.map_sorted_b2s = map_sorted_b2s

    @classmethod
    def _get_mesh_obj(cls, map_mesh_matched, obj_parent, mesh_value):
        import bpy
        if mesh_value in map_mesh_matched:
            return map_mesh_matched[mesh_value]
        mesh_found = None
        for mesh in bpy.data.meshes:
            if mesh.name == mesh_value:
                mesh_found = mesh 
                break
        obj_found = None
        for obj in bpy.data.objects:
            if obj.data == mesh:
                obj_found = obj 
                break 
        print(mesh_found, obj_found)
        if obj_found == obj_parent:
            map_mesh_matched[mesh_value] = mesh_found
            return mesh_found
        return None

    @classmethod
    def _get_shape_key(cls, index_sk, mesh_found):
        keys_blocks = mesh_found.shape_keys.key_blocks
        return keys_blocks[index_sk]
        
    @classmethod
    def _check_vrm0_Vrm0HumanoidPropertyGroup(cls, bpy_data, vae, vrm_info):
        vrm0  = vae.vrm0
        humanoid = vrm0.humanoid

        human_bones = humanoid.human_bones
        bone_names = human_bones.keys()
        print("human_bones in vrm0", list(bone_names), len(bone_names))

        #right_eye, left_eye = None, None 
        right_eye_value, left_eye_value = None, None
        bone_bone_names = []
        for name, bone in human_bones.items():
            print(len(bone_bone_names), name, bone, bone.bone)
            bone_bone_names.append(bone.bone)
            if bone.bone == "leftEye":
                #left_eye = bone 
                left_eye_value = bone.node.value
            elif bone.bone == "rightEye":
                #right_eye = bone
                right_eye_value = bone.node.value

        print("human_bones in vrm0 (bonebone)", list(bone_bone_names), len(bone_bone_names))

        print("right_eye_value", right_eye_value)
        print("left_eye_value", left_eye_value)

        vrm_info.bone_names_in_vrm0 = bone_bone_names
        cls._fill_vrm_eye_bones(bpy_data, left_eye_value, right_eye_value, vrm_info)

    @classmethod
    def _fill_vrm_eye_bones(cls, bpy_data, left_eye_value, right_eye_value, vrm_info):
        from bld_utils_model.finder_bones_eye import FinderBonesEye
        right_eye_bone, left_eye_bone, right_eye_pose_bone, left_eye_pose_bone = FinderBonesEye.find_eye_bones(bpy_data, left_eye_value, right_eye_value)
        
        vrm_info.eye_left_value_in_vrm0 = left_eye_value
        vrm_info.eye_right_value_in_vrm0 = right_eye_value 
        vrm_info.eye_left_bone_in_vrm0 = left_eye_bone
        vrm_info.eye_right_bone_in_vrm0 = right_eye_bone 
        vrm_info.eye_left_pose_bone_in_vrm0 = left_eye_pose_bone
        vrm_info.eye_right_pose_bone_in_vrm0 = right_eye_pose_bone 
