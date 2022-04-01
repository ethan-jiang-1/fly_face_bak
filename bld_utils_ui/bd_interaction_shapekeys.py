from bld_utils_ui.bd_interaction import BdInteraction


class BdInteractionShapeKeys(BdInteraction):
    @classmethod
    def get_shapekeys_info(cls, bpy_data, debug=True):
        from bld_utils_model.finder_shapekeys_info import FinderShapekeysInfo
        return FinderShapekeysInfo.find_grouped_shapekeys_info(bpy_data)

    @classmethod
    def update_kb_value_unified(cls, kb_block, kidx, x_unified):
        if x_unified == 0.0:
            val = x_unified
        else:
           smin = kb_block[kidx].slider_min
           smax = kb_block[kidx].slider_max
           delta = smax - smin 
           val = delta * x_unified + smin 
        kb_block[kidx].value = val        

    @classmethod
    def play_shapekeys_all(cls, renv, shapekeys_info, val_range=None):
        lst_kb_meshes_key_blocks_pairs = cls.find_all_mesh_shapekeys_key_blocks(shapekeys_info)

        if val_range is None or len(val_range) == 0:
            val_range = [0.1, 0.25, 0.50, 0.75, 1.0, 0.15, 0.0]
            if len(lst_kb_meshes_key_blocks_pairs) > 0:
                _, kb_key_names = lst_kb_meshes_key_blocks_pairs[0]
                if len(kb_key_names) > 54:
                    val_range = [0.2, 1.0, 0.0]
                elif len(kb_key_names) > 34:
                    val_range = [0.1, 0.5, 1.0, 0.0]

        for gidx, val in enumerate(lst_kb_meshes_key_blocks_pairs):
            kb_meshes_shapekeys, kb_key_names = val
            cls.send_dev_msg(renv, "shape_key_group{:02}".format(gidx))
            for kidx, key_name in enumerate(kb_key_names):
                cls.send_dev_msg(renv, "{:02} {}".format(kidx, key_name))

                for x in val_range:
                    # all blend shape will be covered
                    for kb_block in kb_meshes_shapekeys:
                        cls.update_kb_value_unified(kb_block, kidx, x)
                    
                    cls.refresh_screen()
        return True

    @classmethod
    def play_shapekeys(cls, renv, shapekeys_info, val_range=None, sel_key_names=[]):
        lst_kb_meshes_key_blocks_pairs = cls.find_all_mesh_shapekeys_key_blocks(shapekeys_info)

        if val_range is None or len(val_range) == 0:
            val_range = [0.10, 0.25, 0.50, 0.75, 1.0, 0.40, 0.0]

        for gidx, val in enumerate(lst_kb_meshes_key_blocks_pairs):
            kb_meshes_shapekeys, kb_key_names = val
            cls.send_dev_msg(renv, "shape_key_group{:02}".format(gidx))
            if len(sel_key_names) == 0:
                selected_key_names = kb_key_names[0:-1:2]
            else:
                selected_key_names = sel_key_names

            for kidx, key_name in enumerate(kb_key_names):
                if key_name in selected_key_names:
                    cls.send_dev_msg(renv, "{:02} {}".format(kidx, key_name))

                    for x in val_range:
                        # all blend shape will be covered
                        for kb_block in kb_meshes_shapekeys:
                            cls.update_kb_value_unified(kb_block, kidx, x)
                        
                        cls.refresh_screen()
        return True

    @classmethod
    def restore_shapekeys(cls, renv, shapekeys_info):
        lst_kb_meshes_key_blocks_pairs = cls.find_all_mesh_shapekeys_key_blocks(shapekeys_info)

        for gidx, val in enumerate(lst_kb_meshes_key_blocks_pairs):
            kb_meshes_shapekeys, kb_key_names = val
            for kidx, key_name in enumerate(kb_key_names):
                for kb_block in kb_meshes_shapekeys:
                    kb_block[kidx].value = 0
                        
        cls.refresh_screen()
        return True

    @classmethod
    def find_all_mesh_shapekeys_key_blocks(cls, shapekeys_info):
        lst_kb_meshes_key_blocks_pairs = []

        if shapekeys_info is not None:
            for _, kmo in shapekeys_info.map_kmo.items():
                kb_meshes_key_blocks = []
                kb_key_names = kmo.lst_key_names
                for (mesh, _) in kmo.lst_mo_pairs:
                    kb_meshes_key_blocks.append(mesh.shape_keys.key_blocks)
                lst_kb_meshes_key_blocks_pairs.append((kb_meshes_key_blocks, kb_key_names))
        return lst_kb_meshes_key_blocks_pairs

    @classmethod
    def find_selected_one_mesh_shapekeys_key_blocks(cls, shapekeys_info, kmo_key=None):
        keys = list(shapekeys_info.map_kmo.keys())
        if len(keys) == 0:
            return None

        if kmo_key is None:
            kmo_key = keys[0]

        if kmo_key not in keys:
            return None

        kb_meshes_key_blocks = []
        kmo = shapekeys_info.map_kmo[kmo_key]
        for (mesh, _) in kmo.lst_mo_pairs:
            kb_meshes_key_blocks.append(mesh.shape_keys.key_blocks)
        return kb_meshes_key_blocks
