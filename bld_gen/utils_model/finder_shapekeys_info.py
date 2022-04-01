from bld_gen.utils_model.easy_dict import EasyDict
import pprint 

class ShapekeysInfo(EasyDict):
    def __init__(self, *args, **kwargs):
        super(ShapekeysInfo, self).__init__(*args, **kwargs)

        self.map_kmo = {}

    def get_dump_bs_txt(self):
        if not self.has_support_data():
            return ""
              
        txt = "\nSHAPEKEYS_INFO:\n"
        map_kmo = self.map_kmo
        txt += "#shape_key(native):\n"
        for key, val in map_kmo.items():
            txt += "  key: {}\n".format(key)
            txt += "  " + pprint.pformat(val, indent=4)
            txt += "\n"
        txt += "\n"
        return txt

    def has_support_data(self):
        map_kmo = self.map_kmo
        if len(map_kmo.keys()) == 0:
            return False
        return True

    def is_valid(self):
        if not self.has_support_data():
            return False
        return True


class FinderShapekeysInfo(object):
    @classmethod
    def short_hash_name(cls, string):
        from hashlib import blake2b
        h = blake2b(digest_size=20)
        h.update(string.encode())
        return h.hexdigest()

    @classmethod
    def _get_kbs_hash_key(cls, key_blocks, kn):
        names = []
        for kb in key_blocks:
            names.append(kb.name)
        lst_key_names = list(names)
        lst_sorted_key_names = sorted(lst_key_names)
        hash_key = cls.short_hash_name(str(lst_sorted_key_names))
        return hash_key, lst_key_names

    @classmethod
    def _find_lst_mesh_obj_with_shape_keys(cls, bpy_data):
        import bpy
        all_objs = {}
        all_meshes = {} 
        lst_mo = []

        objs = bpy_data.objects
        for idx, obj in enumerate(objs):
            if isinstance(obj.data, bpy.types.Mesh):
                all_objs[obj.name] = obj
        print("total of {} objects has mesh associated".format(len(all_objs)))

        meshes = bpy_data.meshes
        for idx, mesh in enumerate(meshes):
            if mesh.shape_keys is not None:
                print(idx, mesh, mesh.name)

                if mesh.shape_keys is not None and mesh.shape_keys.key_blocks is not None:
                    if len(mesh.shape_keys.key_blocks) > 0:
                        for _, obj in all_objs.items():
                            if obj.data == mesh:
                                lst_mo.append((mesh, obj, len(mesh.shape_keys.key_blocks)))
                                all_meshes[mesh.name] = mesh
        print("total of {} mess has obj owner".format(len(all_meshes)))
        return lst_mo

    @classmethod
    def find_all_shapekeys_info(cls, bpy_data):
        lst_mo = cls._find_lst_mesh_obj_with_shape_keys(bpy_data)
        map_skm = {}
        for mo in lst_mo:
            mesh, obj, kn = mo 
            key_blocks = mesh.shape_keys.key_blocks
            for ndx, key in enumerate(key_blocks.keys()):
                if key == "Basis":
                    continue
                if key not in map_skm:
                    map_skm[key] = []
                map_skm[key].append((ndx, key, mesh, obj))
        return map_skm

    @classmethod
    def find_grouped_shapekeys_info(cls, bpy_data):
        lst_mo = cls._find_lst_mesh_obj_with_shape_keys(bpy_data)

        map_kmo = {}
        for mo in lst_mo:
            mesh, obj, kn = mo 
            key_blocks = mesh.shape_keys.key_blocks
            hash_key, lst_key_names = cls._get_kbs_hash_key(key_blocks, kn)
            if hash_key not in map_kmo:
                kmo = EasyDict()
                kmo.lst_key_names = lst_key_names
                kmo.num_key_names = len(lst_key_names)
                kmo.lst_mo_pairs = []
                map_kmo[hash_key] = kmo
            map_kmo[hash_key].lst_mo_pairs.append((mesh, obj))

        shapekey_info = ShapekeysInfo()
        shapekey_info.map_kmo = map_kmo
        if not shapekey_info.is_valid():
            print("ERROR: shapekeys_info is not valid")

        return shapekey_info
