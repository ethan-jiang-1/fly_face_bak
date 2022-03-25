from datetime import datetime
import cv2
import os

class ExpFaceMeshMarker():
    @classmethod
    def mark_face_mesh(cls, mp, face_mesh, image):
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        #drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        #print(drawing_spec)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        d0 = datetime.now()
        results = face_mesh.process(image)
        dt = datetime.now() - d0
        print("inference time {:.3f}".format(dt.total_seconds()))

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            print("len(results.multi_face_landmarks)", len(results.multi_face_landmarks))
            for face_landmarks in results.multi_face_landmarks:
                cls._inpect_landmarks(image, mp_face_mesh, face_landmarks)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            print(image.shape)
        else:
            image = None
        return image, dt

    @classmethod
    def create_face_mesh(cls, mp):
        from utils_inspect.inspect_solution import inspect_solution
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) 
        inspect_solution(face_mesh)
        return face_mesh

    @classmethod
    def close_face_mesh(cls, face_mesh):
        face_mesh.close()

    @classmethod
    def _inpect_landmarks(cls, image, mp_face_mesh, face_landmarks):
        lms = face_landmarks.landmark
        print("len(lms)", len(lms))


def _find_all_images(src_dir):
    if not os.path.isdir(src_dir):
        raise ValueError("{} not folder".format(src_dir))

    names = os.listdir(src_dir)
    filenames = []
    for name in names:
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            filenames.append(os.sep.join([src_dir, name]))
    return filenames

def _make_face_mesh(src_dir, mp=None):
    from utils_inspect.inspect_mp import inspect_mp
    if mp is None:
        import mediapipe as mp
    inspect_mp(mp)

    filenames = _find_all_images(src_dir)

    face_mesh = ExpFaceMeshMarker.create_face_mesh(mp)
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        face_mesh.reset()
        image, _ = ExpFaceMeshMarker.mark_face_mesh(mp, face_mesh, image)
        if image is None:
            print("not able to mark landmark on", filename)

        #img_flip = cv2.flip(image, 1)
        dst_pathname = "{}{}{}".format(os.path.basename(src_dir) + "_output", os.sep, os.path.basename(filename))
        cv2.imwrite(dst_pathname, image)
        print("{} saved".format(dst_pathname))

    ExpFaceMeshMarker.close_face_mesh(face_mesh)
    print("done")

def _get_root_dir():
    import os 

    dir_this = os.path.dirname(__file__)
    dir_root = os.path.dirname(dir_this)
    return dir_root

def _add_root_in_sys_path():
    import sys 

    dir_root = _get_root_dir()
    if dir_root not in sys.path:
        sys.path.append(dir_root)

def do_exp():
    d0 = datetime.now()
    print("loading mediapipe...")
    import mediapipe as mp
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
   
    _make_face_mesh(src_dir, mp)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
