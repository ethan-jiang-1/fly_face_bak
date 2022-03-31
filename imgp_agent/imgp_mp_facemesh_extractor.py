from datetime import datetime
import cv2
import os
import mediapipe as mp
from collections import namedtuple

FME_RESULT = namedtuple('FME_RESULT', "img_org img_facemesh, img_facepaint, mesh_results") 

class ImgpFacemeshExtractor():
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
        print("ImgpFacemeshExtractor inited")

    @classmethod
    def close_imgp(cls):
        if cls.slt_facemesh is not None:
            cls.slt_facemesh.close()
            cls.slt_facemesh = None
        print("ImgpFacemeshExtractor closed")

    @classmethod
    def extract_mesh_features(cls, img_org):
        if cls.slt_facemesh is None:
            cls.init_imgp()

        image, mesh_results = cls._extract_face_mesh(img_org)

        img_facemesh = cls._draw_meshes(image, mesh_results)
        img_facepaint = cls._paint_meshes(image, mesh_results)

        fme_result = FME_RESULT(img_org=img_org,
                                img_facemesh=img_facemesh,
                                img_facepaint=img_facepaint,
                                mesh_results=mesh_results)
        return img_facemesh, fme_result

    @classmethod
    def _extract_face_mesh(cls, img_org):
        image = img_org.copy()

        slt_facemesh = cls.slt_facemesh
        slt_facemesh.reset()

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        d0 = datetime.now()
        mesh_results = slt_facemesh.process(image)
        dt = datetime.now() - d0
        print("inference time {:.3f}".format(dt.total_seconds()))

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, mesh_results

    @classmethod
    def _draw_meshes(cls, image, mesh_results):
        from fmx_mesh.fmx_mesh_grid import FmxMeshGrid
        return FmxMeshGrid.process_img(image, mesh_results)

    @classmethod
    def _paint_meshes(cls, image, mesh_results):
        from fmx_mesh.fmx_mesh_paint import FmxMeshPaint
        return FmxMeshPaint.process_img(image, mesh_results)

    @classmethod
    def _inpect_landmarks(cls, image, face_landmarks):
        lms = face_landmarks.landmark
        print("len(lms)", len(lms))


def _mark_facemesh_imgs(src_dir, selected_names=None):
    from imgp_agent.imgp_common import FileHelper, PlotHelper
    filenames =FileHelper.find_all_images(src_dir)

    ImgpFacemeshExtractor.init_imgp()
    for filename in filenames:
        if selected_names is not None:
            bname = os.path.basename(filename)
            if bname not in selected_names:
                continue 
    
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            print("not image file", filename)
            continue

        image_m, fme_result = ImgpFacemeshExtractor.extract_mesh_features(image)
        if image_m is None:
            print("not able to mark facemesh on", filename)

        FileHelper.save_output_image(image_m, src_dir, filename, "facemesh")
        FileHelper.save_output_image(fme_result.img_facepaint, src_dir, filename, "paintmesh")
        PlotHelper.plot_imgs([image_m, fme_result.img_facepaint])

    ImgpFacemeshExtractor.close_imgp()
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


if __name__ == '__main__':
    _add_root_in_sys_path()
    do_exp()
