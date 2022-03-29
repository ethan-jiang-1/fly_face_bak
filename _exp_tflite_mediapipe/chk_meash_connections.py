
class FaceMeshConnections():
    @classmethod
    def check_mesh_connections(cls, connections):
        cts_ordered = []
        cts = list(connections)

        ct = cts.pop()
        cts_ordered.append(ct)

        while len(cts) != 0:
            vt1 = cts_ordered[-1][1]

            added = False
            for idx, ct in enumerate(cts):
                if ct[0] == vt1 or ct[1] == vt1:
                    cts_ordered.append(cts.pop(idx))
                    added = True
                    break
            if not added:
                vt0= cts_ordered[-1][0]
                for idx, ct in enumerate(cts):
                    if ct[0] == vt0 or ct[1] == vt0:
                        cts_ordered.append(cts.pop(idx))
                        added = True
                        break                
        
            if not added:
                cts_ordered.append((-1, -1))
                cts_ordered.append(cts.pop()) 

        vts = []
        vt = []
        vts.append(vt)
        for ct in cts_ordered:
            ct0 = ct[0]
            if vt == -1:
                vt = []
                vts.append(ct0)
            else:
                vt.append(ct0)
        for vt in vts:
            print(vt)
        return vts


def do_check_mesh_connections():
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    FaceMeshConnections.check_mesh_connections(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

if __name__ == '__main__':
    do_check_mesh_connections()
