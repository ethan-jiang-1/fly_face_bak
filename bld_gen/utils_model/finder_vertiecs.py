from bld_gen.utils_model.easy_dict import EasyDict

VT_VAL_MAX = 10000
VT_VAL_MIN = -VT_VAL_MAX

class FinderVertiecs(object):
    @classmethod
    def find_min_max(cls, mesh):
        edict = EasyDict() 

        vts = mesh.vertices

        lx, ly, lz = VT_VAL_MAX, VT_VAL_MAX, VT_VAL_MAX
        hx, hy, hz = VT_VAL_MIN, VT_VAL_MIN, VT_VAL_MIN
        idx_lx, idx_ly, idx_lz = None, None, None
        idx_hx, idx_hy, idx_hz = None, None, None
        for idx, vt in enumerate(vts):
            if lx > vt.co[0]:
                lx = vt.co[0]
                idx_lx = idx
            if hx < vt.co[0]:
                hx = vt.co[0]
                idx_hx = idx 
            if ly > vt.co[1]:
                ly = vt.co[1]
                idx_ly = idx
            if hy < vt.co[1]:
                hy = vt.co[1]
                idx_hy = idx 
            if lz > vt.co[2]:
                lz = vt.co[2]
                idx_lz = idx
            if hz < vt.co[2]:
                hz = vt.co[2]
                idx_hz = idx

        edict.idx_lx = (idx_lx, lx)
        edict.idx_hx = (idx_hx, hx)
        edict.idx_ly = (idx_ly, ly)
        edict.idx_hy = (idx_hy, hy) 
        edict.idx_lz = (idx_lz, lz) 
        edict.idx_hz = (idx_hz, hz)
        return edict
