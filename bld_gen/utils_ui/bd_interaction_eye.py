import mathutils
from bld_gen.utils_ui.bd_interaction import BdInteraction


class BdInteractionEye(BdInteraction):
    @classmethod
    def play_eye_movement(cls, renv, pose_bone_eye_left, pose_bone_eye_right, degree_range=None):
        if pose_bone_eye_right is None or pose_bone_eye_left is None:
            return False

        cls.send_dev_msg(renv, "EYE movement")
        org_rotation_mode_eye_left = pose_bone_eye_left.rotation_mode
        org_rotation_mode_eye_right = pose_bone_eye_right.rotation_mode

        pose_bone_eye_left.rotation_mode = 'XYZ'
        pose_bone_eye_right.rotation_mode = 'XYZ'

        if degree_range is None or len(degree_range) == 0:
            degree_range = [-5, -3, 3, 5, 0]

        dtr = 3.14/180
        rads = [dg * dtr for dg in degree_range]
        for x in rads:
            for y in rads:
                pose_bone_eye_left.rotation_euler = mathutils.Euler((x, y, 0))
                pose_bone_eye_right.rotation_euler = mathutils.Euler((x, y, 0))
                cls.refresh_screen() 

        pose_bone_eye_left.rotation_mode = org_rotation_mode_eye_left
        pose_bone_eye_right.rotation_mode = org_rotation_mode_eye_right       
        return True
