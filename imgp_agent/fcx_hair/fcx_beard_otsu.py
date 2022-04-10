# import numpy as np
import cv2

try:
    from fcx_base import FcxBase
except:
    from .fcx_base import FcxBase

FMB_FILL_COLOR = (216, 216, 216)

FMB_SELFIE_FILL_COLOR = (192, 192, 192)

class FcxBeardOtsu(FcxBase):
    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        if mesh_results is None or mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        FMB_PX_ALTER = {"U": (0, 0), "R":(0.00, 0), "L":(-0.00, 0), "B":(0, 0.00)}
        image_beard_mask_outter, pt_beard_mask_outter_seed = cls.get_beard_mask_outter(image, mesh_results, FMB_PX_ALTER)
        image_mouth_mask_inner, pt_mouth_mask_inner_seed = cls.get_beard_mouth_inner(image, mesh_results, None)
        img_beard_color0 = cv2.bitwise_and(image, image, mask = image_beard_mask_outter)
        img_beard_color = cv2.bitwise_and(img_beard_color0, img_beard_color0, mask = image_mouth_mask_inner)

        cls.flood_fill(img_beard_color, pt_seed=pt_beard_mask_outter_seed, color_fill=FMB_FILL_COLOR)
        cls.flood_fill(img_beard_color, pt_seed=pt_mouth_mask_inner_seed, color_fill=FMB_FILL_COLOR)

        img_beard_black = cls._filter_beard_to_gray_by_threshold(img_beard_color)
        img_beard_white = cv2.bitwise_not(img_beard_black)
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image_beard_mask_outter, image_mouth_mask_inner, img_beard_color0, img_beard_color, img_beard_white])            

        print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_white

    @classmethod
    def _filter_beard_to_gray_by_threshold(cls, img_beard_color):
        img_beard_color_blur = cv2.blur(img_beard_color, (9,9))
        img_beard_gray = cv2.cvtColor(img_beard_color_blur, cv2.COLOR_BGR2GRAY)

        _, th1 = cv2.threshold(img_beard_gray, 100, 255, cv2.THRESH_BINARY)
        img_beard = th1

        #img_blur = cv2.GaussianBlur(img_beard_gray, (5, 5), 0)
        #_, th3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #img_beard = th3

        return img_beard
