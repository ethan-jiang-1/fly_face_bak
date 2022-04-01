class FinderBonesEye(object):
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
    def find_eye_bones(cls, bpy_data, left_eye_value, right_eye_value):
        right_eye_bone, left_eye_bone = None, None
        right_eye_pose_bone, left_eye_pose_bone = None, None

        armature = cls._get_armature(bpy_data)
        if armature is not None:
            amt_bones = armature.data.bones
            names_bone = []
            for amt_bone in amt_bones:
                names_bone.append(amt_bone.name)
                if amt_bone.name == left_eye_value:
                    left_eye_bone = amt_bone
                elif amt_bone.name == right_eye_value:
                    right_eye_bone = amt_bone
            print("bones", names_bone)

            amt_pose_bones = armature.pose.bones
            names_pose_bone = []
            for amt_pose_bone in amt_pose_bones:
                names_pose_bone.append(amt_pose_bone.name)
                if amt_pose_bone.name == left_eye_value:
                    left_eye_pose_bone = amt_pose_bone
                elif amt_pose_bone.name == right_eye_value:
                    right_eye_pose_bone = amt_pose_bone
            print("pose_bones", names_pose_bone)

        return right_eye_bone, left_eye_bone, right_eye_pose_bone, left_eye_pose_bone
