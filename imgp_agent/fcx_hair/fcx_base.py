import math 
import cv2
import numpy as np


FMB_BEARD_UPPER_L2R = (147, 187, 205, 36, 203, 98, 97, 2, 326, 327, 423, 266, 425, 411, 376)
FMB_BEARD_RIGHT_T2B = (401, 435, 367, 364)
FMB_BEARD_BUTTOM_R2L = (394, 395, 369, 396, 175, 171, 140, 170, 169)
FMB_BEARD_LEFT_B2T = (135, 138, 215, 177)

#FMB_MOUTH_OUTTER = (375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61, 146, 91, 181, 84, 17, 314, 405, 321)

FMB_MOUTH_UPPER_L2R = (39, 37, 0, 267, 269)
FMB_MOUTH_RIGHT_T2B = (270, 409, 291, 375, 321)
FMB_MOUTH_BUTTOM_R2L = (405, 314, 17, 84, 181)
FMB_MOUTH_LEFT_B2T = (91, 146, 61, 185, 40)

#FMB_PX_ALTER = {"U": (0, 0), "R":(0.00, 0), "L":(-0.00, 0), "B":(0, 0.00)}
FMB_FILL_COLOR = (216, 216, 216)

class FcxBase():
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
    def draw_ploypoints_alter(cls, image, face_landmarks, fmv_vertices_with_alter, fmc_color, fmb_px_alter=None):
        image_width, image_height = image.shape[1], image.shape[0]

        contour = []
        llm = face_landmarks.landmark
        for vt in fmv_vertices_with_alter:
            num = vt[0]
            alter_direction = vt[1]

            rx, ry = llm[num].x, llm[num].y,
            if alter_direction is not None: 
                if fmb_px_alter is not None:
                    if alter_direction in fmb_px_alter:
                        dx, dy = fmb_px_alter[alter_direction]
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
    def get_vt_coord(cls, image, face_landmarks, vt):
        image_width, image_height = image.shape[1], image.shape[0]
        llm = face_landmarks.landmark 
        px, py = cls.normalized_to_pixel_coordinates(llm[vt].x, llm[vt].y, image_width, image_height) 
        if px is not None:
            return px, py
        return None      

    @classmethod
    def flood_fill(cls, image, pt_seed=(0, 0), color_fill=FMB_FILL_COLOR):
        h, w = image.shape[:2]
        mask_flood = np.zeros([h+2, w+2],np.uint8)
        cv2.floodFill(image, mask_flood, pt_seed, color_fill, flags=cv2.FLOODFILL_FIXED_RANGE)

    @classmethod
    def get_beard_mouth_inner(cls, image, mesh_results, fmb_px_alter, init_val=(255, 0)):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            return None, None

        image_mouth_mask_inner = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_mouth_mask_inner[:] = init_val[0]

        for face_landmarks in mesh_results.multi_face_landmarks:
            pts_alter = []
            for n in FMB_MOUTH_UPPER_L2R:
                pts_alter.append((n, "U"))
            for n in FMB_MOUTH_RIGHT_T2B:
                pts_alter.append((n, "R"))
            for n in FMB_MOUTH_BUTTOM_R2L:
                pts_alter.append((n, "B"))
            for n in FMB_MOUTH_LEFT_B2T:
                pts_alter.append((n, "L"))

            cls.draw_ploypoints_alter(image_mouth_mask_inner, face_landmarks, pts_alter, init_val[1], fmb_px_alter=fmb_px_alter)

            #cls.draw_ploypoints(image_mouth_mask_inner, face_landmarks, FMB_MOUTH_OUTTER, (0))

        pt_mouth_mask_inner_seed = cls.get_vt_coord(image, face_landmarks, 14)
        return image_mouth_mask_inner, pt_mouth_mask_inner_seed

    @classmethod
    def get_beard_mask_outter(cls, image, mesh_results, fmb_px_alter, init_val=(0, 255)):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            return None, None 
         
        image_beard_mask_outter = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        image_beard_mask_outter[:] = init_val[0]

        for face_landmarks in mesh_results.multi_face_landmarks:
            pts_alter = []
            for n in FMB_BEARD_UPPER_L2R:
                pts_alter.append((n, "U"))
            for n in FMB_BEARD_RIGHT_T2B:
                pts_alter.append((n, "R"))
            for n in FMB_BEARD_BUTTOM_R2L:
                pts_alter.append((n, "B"))
            for n in FMB_BEARD_LEFT_B2T:
                pts_alter.append((n, "L"))

            cls.draw_ploypoints_alter(image_beard_mask_outter, face_landmarks, pts_alter, init_val[1], fmb_px_alter=fmb_px_alter)

        pt_beard_mask_outter_seed = (0, 0)
        return image_beard_mask_outter, pt_beard_mask_outter_seed

    @classmethod
    def ipc_white_balance_color(cls, image):
        # 读取图像
        r, g, b = cv2.split(image)
        r_avg = cv2.mean(r)[0]
        g_avg = cv2.mean(g)[0]
        b_avg = cv2.mean(b)[0]

        if r_avg != 0.0 and g_avg != 0.0 and b_avg != 0.0:
            # 求各个通道所占增益
            k = (r_avg + g_avg + b_avg) / 3
            kr = k / r_avg
            kg = k / g_avg
            kb = k / b_avg
            r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
            g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
            b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
        image_wb = cv2.merge([b, g, r])
        return image_wb

    @classmethod
    def ipc_equalizeHist_color(cls, image):
        b, g, r = cv2.split(image)
        
        b1 = cv2.equalizeHist(b)
        g1 = cv2.equalizeHist(g)
        r1 = cv2.equalizeHist(r)
        
        output = cv2.merge([b1,g1,r1])
        return output

    @classmethod
    def ipc_sharpen_color(cls, img_src):
        img_blur = cv2.GaussianBlur(img_src, (0, 0), 5)
        img_usm = cv2.addWeighted(img_src, 1.5, img_blur, -0.5, 0)
        return img_usm

    @classmethod
    def ipc_normalize_color(cls, img_src):
        dst = np.zeros_like(img_src)
        cv2.normalize(img_src, dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)   
        return dst

