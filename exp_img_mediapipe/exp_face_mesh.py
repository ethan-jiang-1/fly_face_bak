from datetime import datetime
import os

def _make_face_mesh(src_dir, mp=None, cv2=None):
    if mp is None:
        import mediapipe as mp
    if cv2 is None:
        import cv2 as cv2

    from utils_inspect.inspect_solution import inspect_solution  

    if not os.path.isdir(src_dir):
        raise ValueError("{} not folder".format(src_dir))

    names = os.listdir(src_dir)
    filenames = []
    for name in names:
        if name.endswith(".jpg") or name.endswith(".jpeg"):
            filenames.append(os.sep.join([src_dir, name]))

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # For webcam input:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    print(drawing_spec)
    #cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        inspect_solution(face_mesh)

        for filename in filenames:
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is None:
                print("not image file", filename)
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
            else:
                print("no landmark found for ", filename)

            img_flip = cv2.flip(image, 1)
            dst_pathname = "{}{}{}".format(src_dir + "_output", os.sep, os.path.basename(filename))
            cv2.imwrite(dst_pathname, img_flip)
    print("done")


def do_exp():
    d0 = datetime.now()
    print("loading mediapipe...")
    import mediapipe as mp
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    d0 = datetime.now()
    import cv2 as cv2
    dt = datetime.now() - d0
    print("loading done", "{:.2f}sec".format(dt.total_seconds()))

    src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    _make_face_mesh(src_dir, mp, cv2)

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

if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
