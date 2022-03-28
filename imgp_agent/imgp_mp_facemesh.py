from datetime import datetime
import cv2
import os
import mediapipe as mp
import numpy as np
import math

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

FMV_FACE_OVAL = (127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234)
FMV_LEFT_EYE = (382, 398, 384, 385, 386, 387, 388, 466, 263, 263, 249, 390, 373, 374, 380, 381)
FMV_RIGHT_EYE = (155, 173, 157, 158, 159, 160, 161, 246, 33, 33, 7, 163, 144, 145, 153, 154)
FMV_MOUTH_INNER = (310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311)
FMV_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)
FMV_LEFT_IRIS =(474, 475, 476, 477)
FMV_RIGHT_IRIS = (470, 471, 472, 469)
FMV_LEFT_EYEBROW = (300, 293, 334, 296, 336,  285, 295, 282, 283, 276)
FMV_RIGHT_EYEBROW =(70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
FMV_NOSE = (8, 417, 465,  343, 437, 420, 279, 278, 439, 438, 457, 274, 1, 44, 237, 218, 219, 48, 49, 198, 217, 114, 245, 193)


FMC_FACE_OVAL = (256, 256, 0)
FMC_LEFT_EYE = (0, 0, 32)
FMC_RIGHT_EYE = (0, 0, 32)
FMC_MOUTH_INNER = (0, 0, 64)
FMC_MOUTH_OUTTER = (0, 0, 160)
FMC_LEFT_IRIS = (0, 0, 196)
FMC_RIGHT_IRIS = (0, 0, 196)
FMC_LEFT_EYEBROW = (0, 0, 255)
FMC_RIGHT_EYEBROW = (0, 0, 255)
FMC_NOSE = (0, 0, 128)

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
    def create_face_mesh_core(cls, img_org):
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

        image_with_paint = cls._paint_meshes(image, results)
        image_with_mesh = cls._draw_meshes(image, results)
        return image_with_mesh, image_with_paint

    @classmethod
    def mark_face_mesh(cls, img_org):
        image_with_mesh, image_with_paint = cls.create_face_mesh_core(img_org)
        return image_with_mesh, image_with_paint 

    @classmethod
    def _draw_meshes(cls, image, results):
        if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

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
        return image

    @classmethod
    def _normalized_to_pixel_coordinates(cls, normalized_x, normalized_y, image_width, image_height):
        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None, None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    @classmethod
    def _draw_ploypoints(cls, image, face_landmarks, fmv_vertices, fmc_color):
        image_width, image_height = image.shape[1], image.shape[0]

        contour = []
        llm = face_landmarks.landmark
        for vt in fmv_vertices:
            px, py = cls._normalized_to_pixel_coordinates(llm[vt].x, llm[vt].y, image_width, image_height) 
            if px is not None and py is not None:
                contour.append((px, py))

        np_contour = np.array(contour)
        cv2.fillConvexPoly(image, np_contour, fmc_color) 

    @classmethod
    def _paint_meshes(cls, image, results):
        if results.multi_face_landmarks is None or len(results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        image = np.zeros(image.shape, dtype=image.dtype)
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        print("len(results.multi_face_landmarks)", len(results.multi_face_landmarks))
        for face_landmarks in results.multi_face_landmarks:
            cls._inpect_landmarks(image, mp_face_mesh, face_landmarks)

            cls._draw_ploypoints(image, face_landmarks, FMV_FACE_OVAL, FMC_FACE_OVAL)

            landmark_ds = mp_drawing_styles.DrawingSpec(color=_RED, thickness=1, circle_radius=2)
            connection_ds = mp_drawing_styles.DrawingSpec(color=FMC_FACE_OVAL, thickness=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=landmark_ds,
                connection_drawing_spec=connection_ds)

            cls._draw_ploypoints(image, face_landmarks, FMV_LEFT_EYE, FMC_LEFT_EYE)
            cls._draw_ploypoints(image, face_landmarks, FMV_RIGHT_EYE, FMC_RIGHT_EYE)

            cls._draw_ploypoints(image, face_landmarks, FMV_LEFT_IRIS, FMC_LEFT_IRIS)
            cls._draw_ploypoints(image, face_landmarks, FMV_RIGHT_IRIS, FMC_RIGHT_IRIS)

            cls._draw_ploypoints(image, face_landmarks, FMV_MOUTH_OUTTER, FMC_MOUTH_OUTTER)
            cls._draw_ploypoints(image, face_landmarks, FMV_MOUTH_INNER, FMC_MOUTH_INNER)

            cls._draw_ploypoints(image, face_landmarks, FMV_LEFT_EYEBROW, FMC_LEFT_EYEBROW)
            cls._draw_ploypoints(image, face_landmarks, FMV_RIGHT_EYEBROW, FMC_RIGHT_EYEBROW)
            cls._draw_ploypoints(image, face_landmarks, FMV_NOSE, FMC_NOSE)

            # ds_tesselation = mp_drawing_styles.DrawingSpec(color=_GRAY, thickness=1)
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=ds_tesselation)
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
        print(image.shape, image.dtype)
        return image

    @classmethod
    def _inpect_landmarks(cls, image, mp_face_mesh, face_landmarks):
        lms = face_landmarks.landmark
        print("len(lms)", len(lms))


def _mark_facemesh_imgs(src_dir, selected_names=None):
    from imgp_agent.imgp_common import FileHelper, PlotHelper
    filenames =FileHelper.find_all_images(src_dir)

    ImgpFacemeshMarker.init_imgp()
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue 
    
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image_m, image_p = ImgpFacemeshMarker.mark_face_mesh(image)
        if image_m is None:
            print("not able to mark facemesh on", filename)

        FileHelper.save_output_image(image_m, src_dir, filename, "facemesh")
        FileHelper.save_output_image(image_p, src_dir, filename, "paintmesh")
        PlotHelper.plot_img(image_p)

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

    #selected_names = None
    #selected_names = ["hsi_image1.jpeg"]
    #selected_names = ["hsi_image4.jpeg"]
    selected_names = ["icl_image5.jpeg"]
       
    _mark_facemesh_imgs(src_dir, selected_names=selected_names)


def do_check_mesh_connections():
    mp_face_mesh = mp.solutions.face_mesh
    FaceMeshConnections.check_mesh_connections(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)

if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
    #do_check_mesh_connections()
