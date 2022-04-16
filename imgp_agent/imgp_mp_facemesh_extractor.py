from datetime import datetime
import cv2
import os
import mediapipe as mp
#import copy
from collections import namedtuple

FME_RESULT = namedtuple('FME_RESULT', "img_org img_facemesh img_facepaint img_outline mesh_results") 

class ImgpFacemeshExtractor():
    #slt_facemesh = None
    #slt_reference = 0 

    @classmethod
    def init_imgp(cls):
        # cls.slt_reference += 1
        # if cls.slt_facemesh is None:
        #     #from utils_inspect.inspect_solution import inspect_solution
        #     mp_face_mesh = mp.solutions.face_mesh
        #     d0 = datetime.now()
        #     cls.slt_facemesh = mp_face_mesh.FaceMesh(
        #         max_num_faces=1,
        #         refine_landmarks=True,
        #         min_detection_confidence=0.5,
        #         min_tracking_confidence=0.5) 
        #     #inspect_solution(cls.slt_facemesh)
        #     dt = datetime.now() - d0
        #     print("dt", dt.total_seconds())
        print("#ImgpFacemeshExtractor inited")

    @classmethod
    def close_imgp(cls):
        # cls.slt_reference -= 1
        # if cls.slt_reference == 0:
        #     if cls.slt_facemesh is not None:
        #         cls.slt_facemesh.close()
        #         if cls.slt_facemesh is not None:
        #             del cls.slt_facemesh
        #         cls.slt_facemesh = None
        print("#ImgpFacemeshExtractor closed")
        
    @classmethod
    def extract_mesh_features(cls, img_org, debug=False):
        d0 = datetime.now()
        image_bgr, mesh_results = cls._extract_face_mesh(img_org, debug=debug)

        img_facemesh = cls._draw_meshes(image_bgr, mesh_results)
        img_facepaint = cls._paint_meshes(image_bgr, mesh_results)
        img_outline = cls._paint_outline(image_bgr, mesh_results)
        #img_facemesh = None 
        #img_facepaint = None 
        #img_outline = None 
        dt = datetime.now() - d0
        print("inference time(fm) :{:.3f}".format(dt.total_seconds()))

        fme_result = FME_RESULT(img_org=img_org,
                                img_facemesh=img_facemesh,
                                img_facepaint=img_facepaint,
                                img_outline = img_outline,
                                mesh_results=mesh_results)
        return img_facemesh, fme_result

    @classmethod
    def _extract_face_mesh(cls, img_org, debug=False):
        image = img_org.copy()

        #slt_facemesh = cls.slt_facemesh
        #slt_facemesh.reset()
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.0) as slt_facemesh:

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            d0 = datetime.now()
            mesh_results = slt_facemesh.process(image_rgb)
            dt = datetime.now() - d0
            if debug:
                print("inference time(fm): {:.3f}".format(dt.total_seconds()))

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            #del mesh_results
        return image_bgr, mesh_results
        #return image_bgr, None

    @classmethod
    def _draw_meshes(cls, image, mesh_results):
        try:
            from fmx_face.fmx_face_mesh import FmxFaceMesh
        except:
            from .fmx_face.fmx_face_mesh import FmxFaceMesh            
        return FmxFaceMesh.process_img(image, mesh_results)

    @classmethod
    def _paint_meshes(cls, image, mesh_results):
        try:
            from fmx_face.fmx_face_paint import FmxFacePaint
        except:
            from .fmx_face.fmx_face_paint import FmxFacePaint
        return FmxFacePaint.process_img(image, mesh_results)

    @classmethod
    def _paint_outline(cls, image, mesh_results):
        try:
            from fmx_face.fmx_face_outline import FmxFaceOutline
        except:
            from .fmx_face.fmx_face_outline import FmxFaceOutline
        return FmxFaceOutline.process_img(image, mesh_results)

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
