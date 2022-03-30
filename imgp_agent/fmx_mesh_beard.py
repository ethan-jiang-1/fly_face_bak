import numpy as np
import math
import cv2

_RED = (48, 48, 255)
_GREEN = (48, 255, 48)
_BLUE = (192, 101, 21)
_YELLOW = (0, 204, 255)
_GRAY = (128, 128, 128)
_PURPLE = (128, 64, 128)
_PEACH = (180, 229, 255)
_WHITE = (224, 224, 224)

FMB_UPPER_L2R = (147, 187, 205, 36, 203, 98, 97, 2, 326, 327, 423, 266, 425, 411, 376)
FMB_RIGHT_T2B = (401, 435, 367, 364)
FMB_BUTTOM_R2L = (394, 395, 369, 396, 175, 171, 140, 170, 169)
FMB_LEFT_B2T = (135, 138, 215, 177)

FMB_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)

FF_FILL_COLOR = (216, 216, 216)

class FmxMeshBeard():
    @classmethod
    def normalized_to_pixel_coordinates(cls, normalized_x, normalized_y, image_width, image_height):
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
    def draw_ploypoints(cls, image, face_landmarks, fmv_vertices, fmc_color):
        image_width, image_height = image.shape[1], image.shape[0]

        contour = []
        llm = face_landmarks.landmark
        for vt in fmv_vertices:
            px, py = cls.normalized_to_pixel_coordinates(llm[vt].x, llm[vt].y, image_width, image_height) 
            if px is not None and py is not None:
                contour.append([px, py])

        np_contour = np.array(contour)
        #cv2.polylines(image, [np_contour], 10, fmc_color)
        cv2.fillPoly(image, [np_contour], fmc_color) 

    @classmethod
    def _get_vt_coord(cls, image, face_landmarks, vt):
        image_width, image_height = image.shape[1], image.shape[0]
        llm = face_landmarks.landmark 
        px, py = cls.normalized_to_pixel_coordinates(llm[vt].x, llm[vt].y, image_width, image_height) 
        if px is not None:
            return px, py
        return None       

    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        if mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        image_mask_outter, image_mask_inner, pt_mouth_inner = cls._get_beard_masks(image, mesh_results)
        img_bread_color0 = cv2.bitwise_and(image, image, mask = image_mask_outter)
        img_bread_color = cv2.bitwise_and(img_bread_color0, img_bread_color0, mask = image_mask_inner)

        cls._flood_fill(img_bread_color, pt_seed=(0, 0), color_fill=FF_FILL_COLOR)
        if pt_mouth_inner is not None:
            # color_pt_month_inner = img_bread_color[pt_mouth_inner[1], pt_mouth_inner[0]]
            # if np.not_equal(np.array(FF_FILL_COLOR, dtype=np.uint8).all(), color_pt_month_inner.all()):
            cls._flood_fill(img_bread_color, pt_seed=pt_mouth_inner, color_fill=FF_FILL_COLOR)

        img_bread = cls._filter_bread(img_bread_color)
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image_mask_outter, image_mask_inner, img_bread_color0, img_bread_color, img_bread])            

        print(img_bread.shape, img_bread.dtype)
        return img_bread

    @classmethod
    def _get_beard_masks(cls, image, mesh_results):
        image_mask_outter = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_mask_inner = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_mask_inner[:] = (255)
        print("len(mesh_results.multi_face_landmarks)", len(mesh_results.multi_face_landmarks))
        pt_mouth_inner = None
        for face_landmarks in mesh_results.multi_face_landmarks:
            pts = []
            pts.extend(FMB_UPPER_L2R)
            pts.extend(FMB_RIGHT_T2B)
            pts.extend(FMB_BUTTOM_R2L)
            pts.extend(FMB_LEFT_B2T)

            cls.draw_ploypoints(image_mask_outter, face_landmarks, pts, (255))

            cls.draw_ploypoints(image_mask_inner, face_landmarks, FMB_MOUTH_OUTTER, (0))

            pt_mouth_inner = cls._get_vt_coord(image, face_landmarks, 14)
        return image_mask_outter, image_mask_inner, pt_mouth_inner

    @classmethod
    def _flood_fill(cls, image, pt_seed=(0, 0), color_fill=FF_FILL_COLOR):
        h, w = image.shape[:2]
        mask_flood = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(image, mask_flood, pt_seed, color_fill, flags=cv2.FLOODFILL_FIXED_RANGE)

    @classmethod
    def _filter_bread(cls, img_bread_color):
        img_bread_gray = cv2.cvtColor(img_bread_color, cv2.COLOR_BGR2GRAY)

        _, th1 = cv2.threshold(img_bread_gray, 100, 255, cv2.THRESH_BINARY)

        #img_blur = cv2.GaussianBlur(img_bread_gray, (5, 5), 0)
        #_, th3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_bread = th1
        return img_bread
