import numpy as np
import cv2

try:
    from fcx_base import FcxBase
except:
    from .fcx_base import FcxBase


FMB_UPPER_L2R = (147, 187, 205, 36, 203, 98, 97, 2, 326, 327, 423, 266, 425, 411, 376)
FMB_RIGHT_T2B = (401, 435, 367, 364)
FMB_BUTTOM_R2L = (394, 395, 369, 396, 175, 171, 140, 170, 169)
FMB_LEFT_B2T = (135, 138, 215, 177)

FMB_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)

FMB_FILL_COLOR = (216, 216, 216)

FMB_SELFIE_FILL_COLOR = (192, 192, 192)

class FcxBeardOtsu(FcxBase):
    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        if mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        image_beard_mask_outter, pt_beard_mask_outter_seed = cls._get_beard_mask_outter(image, mesh_results)
        image_mouth_mask_inner, pt_mouth_mask_inner_seed = cls._get_beard_mouth_inner(image, mesh_results)
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
    def _get_beard_mask_outter(cls, image, mesh_results):
        image_beard_mask_outter = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for face_landmarks in mesh_results.multi_face_landmarks:
            pts = []
            for n in FMB_UPPER_L2R:
                pts.append((n, "U"))
            for n in FMB_RIGHT_T2B:
                pts.append((n, "R"))
            for n in FMB_BUTTOM_R2L:
                pts.append((n, "B"))
            for n in FMB_LEFT_B2T:
                pts.append((n, "L"))

            cls.draw_ploypoints_alter(image_beard_mask_outter, face_landmarks, pts, (255))

        pt_beard_mask_outter_seed = (0, 0)
        return image_beard_mask_outter, pt_beard_mask_outter_seed

    @classmethod
    def _get_beard_mouth_inner(cls, image, mesh_results):
        image_mouth_mask_inner = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_mouth_mask_inner[:] = (255)
        pt_mouth_mask_inner_seed = None
        for face_landmarks in mesh_results.multi_face_landmarks:
            cls.draw_ploypoints_alter(image_mouth_mask_inner, face_landmarks, FMB_MOUTH_OUTTER, (0))

        pt_mouth_mask_inner_seed = cls.get_vt_coord(image, face_landmarks, 14)
        return image_mouth_mask_inner, pt_mouth_mask_inner_seed

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
