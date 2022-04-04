import cv2 

try:
    from fmx_face_paint import FmxFacePaint
except:
    from .fmx_face_paint import FmxFacePaint

class FmxFaceOutline(FmxFacePaint):
    @classmethod
    def process_img(cls, image, mesh_results):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            return None
        img_outline = cls.paint_face_core(image, mesh_results, paint_mode="outline")
        img_outline_gray = cv2.cvtColor(img_outline, cv2.COLOR_RGB2GRAY)
        return img_outline_gray
