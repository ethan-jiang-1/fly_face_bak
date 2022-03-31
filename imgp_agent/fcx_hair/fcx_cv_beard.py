import numpy as np
import math
import cv2

# _RED = (48, 48, 255)
# _GREEN = (48, 255, 48)
# _BLUE = (192, 101, 21)
# _YELLOW = (0, 204, 255)
# _GRAY = (128, 128, 128)
# _PURPLE = (128, 64, 128)
# _PEACH = (180, 229, 255)
# _WHITE = (224, 224, 224)

FMB_UPPER_L2R = (147, 187, 205, 36, 203, 98, 97, 2, 326, 327, 423, 266, 425, 411, 376)
FMB_RIGHT_T2B = (401, 435, 367, 364)
FMB_BUTTOM_R2L = (394, 395, 369, 396, 175, 171, 140, 170, 169)
FMB_LEFT_B2T = (135, 138, 215, 177)

FMB_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)

FMB_FILL_COLOR = (216, 216, 216)

FMB_PX_ALTER = {"U": (0, 0), "R":(0.00, 0), "L":(-0.00, 0), "B":(0, 0.00)}

FMB_SELFIE_FILL_COLOR = (192, 192, 192)

class FcxCvBeard():
    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        image_flood = image.copy()
        cls._flood_fill(image_flood, pt_seed=(0,0), color_fill=FMB_SELFIE_FILL_COLOR)

        ksize = 11  # kernal size
        lamda = np.pi / 2.0  # length of wave
        theta = np.pi * 0.5  # 90 degree
        #theta = np.pi * 0.0  # 0 degree
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        gabor_filter = kern 

        accum = np.zeros_like(image_flood)
        img_fimg = cv2.filter2D(image_flood, cv2.CV_8UC1, gabor_filter)
        img_accum = np.maximum(accum, img_fimg, accum)

        img_beard_color_blur = cv2.blur(img_accum, (9,9))
        img_beard_gray = cv2.cvtColor(img_beard_color_blur, cv2.COLOR_BGR2GRAY)
        
        img_beard_black = img_beard_gray
        img_beard_white = cv2.bitwise_not(img_beard_black)

        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image, image_flood, img_fimg, img_accum, img_beard_black, img_beard_white])            

        print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_white

    @classmethod
    def process_img_old(cls, image, mesh_results, debug=False):
        if mesh_results.multi_face_landmarks is None or len(mesh_results.multi_face_landmarks) == 0:
            print("no face_landmarks found")
            return None 

        image_beard_mask_outter, pt_beard_mask_outter_seed = cls._get_beard_mask_outter(image, mesh_results)
        image_mouth_mask_inner, pt_mouth_mask_inner_seed = cls._get_beard_mouth_inner(image, mesh_results)
        img_beard_color0 = cv2.bitwise_and(image, image, mask = image_beard_mask_outter)
        img_beard_color = cv2.bitwise_and(img_beard_color0, img_beard_color0, mask = image_mouth_mask_inner)

        cls._flood_fill(img_beard_color, pt_seed=pt_beard_mask_outter_seed, color_fill=FMB_FILL_COLOR)
        cls._flood_fill(img_beard_color, pt_seed=pt_mouth_mask_inner_seed, color_fill=FMB_FILL_COLOR)

        img_beard_black = cls._filter_beard_to_gray_by_threshold(img_beard_color)
        img_beard_white = cv2.bitwise_not(img_beard_black)
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image_beard_mask_outter, image_mouth_mask_inner, img_beard_color0, img_beard_color, img_beard_white])            

        print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_white

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
    def draw_ploypoints_alter(cls, image, face_landmarks, fmv_vertices, fmc_color):
        image_width, image_height = image.shape[1], image.shape[0]

        contour = []
        llm = face_landmarks.landmark
        for vt in fmv_vertices:
            num = vt
            alter_direction = None 
            if hasattr(vt, "__len__"):
                num = vt[0]
                alter_direction = vt[1]

            rx, ry = llm[num].x, llm[num].y,
            if alter_direction is not None: 
                if alter_direction in FMB_PX_ALTER:
                    dx, dy = FMB_PX_ALTER[alter_direction]
                    rx += dx 
                    ry += dy
                else:
                    print("unknown direction", alter_direction)

            px, py = cls.normalized_to_pixel_coordinates(rx, ry,  image_width, image_height) 
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

        pt_mouth_mask_inner_seed = cls._get_vt_coord(image, face_landmarks, 14)
        return image_mouth_mask_inner, pt_mouth_mask_inner_seed

    @classmethod
    def _flood_fill(cls, image, pt_seed=(0, 0), color_fill=FMB_FILL_COLOR):
        h, w = image.shape[:2]
        mask_flood = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(image, mask_flood, pt_seed, color_fill, flags=cv2.FLOODFILL_FIXED_RANGE)

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
