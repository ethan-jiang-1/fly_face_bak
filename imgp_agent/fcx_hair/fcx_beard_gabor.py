import numpy as np
import math
import cv2

FMB_FILL_COLOR = (216, 216, 216)
FMB_SELFIE_FILL_COLOR = (192, 192, 192)
FMB_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)


class FcxBeardGabor():
    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        img_beard_region = cls._clean_region(image, mesh_results)

        img_gabor_filtered = cls._filter_by_gabor_filter(img_beard_region)

        img_beard_color_blur = cv2.blur(img_gabor_filtered, (5,5))
        img_beard_gray = cv2.cvtColor(img_beard_color_blur, cv2.COLOR_BGR2GRAY)
        
        if cls._has_beard_at_keypoints(img_beard_gray, mesh_results):
            _, img_beard_gray_th = cv2.threshold(img_beard_gray, 255-16, 255, cv2.THRESH_BINARY)
            img_beard_black = img_beard_gray_th
        else:
            img_beard_black = np.zeros_like(img_beard_gray)
        img_beard_white = cv2.bitwise_not(img_beard_black)

        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image, img_beard_region, img_gabor_filtered, img_beard_black, img_beard_white],
                                 names=["org", "flood", "gabor", "black", "white"])            

        print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_white

    @classmethod
    def _has_beard_at_keypoints(cls, img_beard_gray, mesh_results):
        vts = [165, 167, 164, 393, 391, 182, 18, 406, 200, 199]
        face_landmarks = mesh_results.multi_face_landmarks[0]

        COLOR_HAS_BEARD = 255 - 16
        beard_cnt = 0
        beard_vals = []
        for vt in vts:
            pt = cls._get_vt_coord(img_beard_gray, face_landmarks, vt)
            if pt is None:
                continue
            pt_x, pt_y = pt
            val = int(img_beard_gray[pt_y, pt_x])
            beard_vals.append(val)
            if val < COLOR_HAS_BEARD:
                beard_cnt += 1
        if beard_cnt >= 5:
            return True
        val_mean = np.array(beard_vals).mean()
        if val_mean < COLOR_HAS_BEARD + 8:
            return True
        return False

    @classmethod
    def _clean_region(cls, image, mesh_results):
        image_flood = image.copy()
        cls._flood_fill(image_flood, pt_seed=(0,0), color_fill=FMB_SELFIE_FILL_COLOR)
  
        image_mouth_mask_inner, pt_mouth_mask_inner_seed = cls._get_beard_mouth_inner(image, mesh_results)
        img_beard_color = cv2.bitwise_and(image_flood, image_flood, mask = image_mouth_mask_inner)
        cls._flood_fill(img_beard_color, pt_seed=pt_mouth_mask_inner_seed, color_fill=FMB_FILL_COLOR)

        face_landmarks = mesh_results.multi_face_landmarks[0]
        pt_nose = cls._get_vt_coord(image, face_landmarks, 4)
        nose_y = pt_nose[1]
        image_witdh = image.shape[1]
        pts = [(0, 0), (image_witdh, 0), (image_witdh, nose_y), (0, nose_y)]
        cv2.fillConvexPoly(img_beard_color, np.array(pts), FMB_FILL_COLOR)   

        return img_beard_color

    @classmethod
    def _filter_by_gabor_filter(cls, image):
        ksize = 11  # kernal size
        lamda = np.pi / 2.0  # length of wave
        theta = np.pi * 0.5  # 90 degree
        #theta = np.pi * 0.0  # 0 degree
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        gabor_filters = kern 

        img_accum = np.zeros_like(image)
        for gabor_filter in gabor_filters:
            img_fimg = cv2.filter2D(image, cv2.CV_8UC1, gabor_filter)
            img_accum = np.maximum(img_accum, img_fimg, img_accum)
        return img_accum

    @classmethod
    def _flood_fill(cls, image, pt_seed=(0, 0), color_fill=FMB_FILL_COLOR):
        h, w = image.shape[:2]
        mask_flood = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(image, mask_flood, pt_seed, color_fill, flags=cv2.FLOODFILL_FIXED_RANGE)

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
            num = vt
            rx, ry = llm[num].x, llm[num].y,
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
    def _get_beard_mouth_inner(cls, image, mesh_results):
        image_mouth_mask_inner = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_mouth_mask_inner[:] = (255)
        pt_mouth_mask_inner_seed = None
        for face_landmarks in mesh_results.multi_face_landmarks:
            cls.draw_ploypoints(image_mouth_mask_inner, face_landmarks, FMB_MOUTH_OUTTER, (0))

        pt_mouth_mask_inner_seed = cls._get_vt_coord(image, face_landmarks, 14)
        return image_mouth_mask_inner, pt_mouth_mask_inner_seed
