from bld_utils_ui.bd_interaction import BdInteraction


class BdInteractionVrm(BdInteraction):
    @classmethod
    def get_vrm_info(cls, bpy_data, debug=True):
        from bld_utils_model.finder_vrm_info import FinderVrmInfo
        return FinderVrmInfo.find_info(bpy_data)

    @classmethod
    def play_vrm_blendshape_all(cls, renv, vrm_info, bs_strength=0):
        map_sorted_b2s = vrm_info.map_sorted_b2s

        fval = 0 
        if bs_strength > 0 and bs_strength <= 1.0:
            fval = bs_strength
        
        vals = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 0.5, fval]
        if len(map_sorted_b2s.keys()) > 30:
            vals = [0.0, 0.5, 1.0, fval]

        for bsname, skis in map_sorted_b2s.items():
            keys = "{}: ".format(bsname)
            for ski in skis:
                index_sk = ski[0]
                keys += "{} ".format(index_sk)
            cls.send_dev_msg(renv, keys)
            cls.refresh_screen()

            for x in vals:
                keys = "{}: ".format(bsname)
                for ski in skis:
                    index_sk = ski[0]
                    weight_b2s = ski[1]
                    #mesh_value = ski[2]
                    shape_key = ski[3]
                    keys += "{} ".format(index_sk)

                    shape_key.value = x * weight_b2s
                    cls.refresh_screen()

        return True

    @classmethod
    def play_vrm_eye_movement(cls, renv, vrm_info):
        from bld_utils_ui.bd_interaction_eye import BdInteractionEye

        pose_bone_eye_left = vrm_info.eye_left_pose_bone_in_vrm0
        pose_bone_eye_right = vrm_info.eye_right_pose_bone_in_vrm0

        return BdInteractionEye.play_eye_movement(renv, pose_bone_eye_left, pose_bone_eye_right)
