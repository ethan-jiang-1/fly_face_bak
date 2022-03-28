from datetime import datetime
import cv2
import os
import mediapipe as mp

class ImgpFacemeshMarker():
    slt_facemesh = None 

    @classmethod
    def init_imgp(cls):
        if cls.slt_facemesh is None:
            from utils_inspect.inspect_solution import inspect_solution
            mp_face_mesh = mp.solutions.face_mesh
            cls.slt_facemesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) 
            inspect_solution(cls.slt_facemesh)
        print("ImgpFacemeshMarker inited")

    @classmethod
    def close_imgp(cls):
        if cls.slt_facemesh is not None:
            cls.slt_facemesh.close()
            cls.slt_facemesh = None
        print("ImgpFacemeshMarker closed")

    @classmethod
    def mark_face_mesh(cls, img_org, paint_region=False):
        if cls.slt_facemesh is None:
            cls.init_imgp()

        image = img_org.copy()

        slt_facemesh = cls.slt_facemesh
        slt_facemesh.reset()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        d0 = datetime.now()
        results = slt_facemesh.process(image)
        dt = datetime.now() - d0
        print("inference time {:.3f}".format(dt.total_seconds()))

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if paint_region:
            image_with_paint = cls._paint_meshes(image, results)
            return image_with_paint, None 
        else:
            image_with_mesh = cls._draw_meshes(image, results)
            return image_with_mesh, None 

    @classmethod
    def _draw_meshes(cls, image, results):
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
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
            print(image.shape, image.dtype)
        else:
            image = None
        return image

    @classmethod
    def _paint_meshes(cls, image, results):
        return image

    @classmethod
    def _inpect_landmarks(cls, image, mp_face_mesh, face_landmarks):
        lms = face_landmarks.landmark
        print("len(lms)", len(lms))

def _mark_facemesh_imgs(src_dir):
    from imgp_agent.imgp_common import FileOp
    filenames =FileOp.find_all_images(src_dir)

    ImgpFacemeshMarker.init_imgp()
    for filename in filenames:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image, _ = ImgpFacemeshMarker.mark_face_mesh(image)
        if image is None:
            print("not able to mark facemesh on", filename)

        FileOp.save_output_image(image, src_dir, filename, "facemesh")

    ImgpFacemeshMarker.close_imgp()
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
    #src_dir = os.sep.join([_get_root_dir(), "_test_imgs_1"])
    #src_dir = os.sep.join([_get_root_dir(), "hsi_tflite_interpeter", "_reserved_imgs"])
    src_dir = os.sep.join([_get_root_dir(), "utils_inspect", "_sample_imgs"])
   
    _mark_facemesh_imgs(src_dir)


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
