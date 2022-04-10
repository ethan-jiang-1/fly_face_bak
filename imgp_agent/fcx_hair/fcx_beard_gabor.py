import numpy as np
import cv2

FMB_FILL_COLOR = (216, 216, 216)
FMB_SELFIE_FILL_COLOR = (192, 192, 192)

try:
    from fcx_base import FcxBase
except:
    from .fcx_base import FcxBase

class FcxBeardGabor(FcxBase):
    @classmethod
    def process_img(cls, image, mesh_results, debug=False):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            return None 
            
        img_beard_region = cls._clean_region(image, mesh_results)
        img_beard_region_wb = cls.ipc_white_balance_color(img_beard_region)
        #img_beard_region_eh = cls.ipc_equalizeHist_color(img_beard_region)
        img_beard_region_sp = cls.ipc_sharpen_color(img_beard_region_wb)

        img_gabor_filtered = cls._filter_by_gabor_filter(img_beard_region_sp)
        img_gabor_filtered_no_mouth = cls._clean_region_mouth_inner(img_gabor_filtered, mesh_results, color_fill=(255,255,255))

        img_beard_gabor_color_blur = cv2.blur(img_gabor_filtered_no_mouth, (3, 3))
        img_beard_gabor_gray = cv2.cvtColor(img_beard_gabor_color_blur, cv2.COLOR_BGR2GRAY)
        
        img_box_mouth = cls._get_box_img_around_mouth(img_beard_gabor_gray, mesh_results)
        has_beard, val_has_beard = cls._has_beard_at_keypoints(img_beard_gabor_gray, mesh_results, img_box_mouth)
        if has_beard:
            _, img_beard_gray_th = cv2.threshold(img_beard_gabor_gray, val_has_beard, 255, cv2.THRESH_BINARY)
            img_beard_black = img_beard_gray_th
        else:
            img_beard_black = np.zeros_like(img_beard_gabor_gray)
        img_beard_white = cv2.bitwise_not(img_beard_black)

        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image, img_beard_region,img_beard_region_sp, img_gabor_filtered, img_beard_gabor_gray, img_box_mouth, img_beard_black, img_beard_white],
                                 names=["org", "flood", "sharpen", "gabor", "gabor_gray", "gabor_box", "black", "white"])            

        print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_white

    @classmethod
    def _get_beard_avg_val(cls, img_beard_gray, pt):
        try:
            vals = []
            pt_x, pt_y = pt
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    vals.append(float(int(img_beard_gray[pt_y + dy, pt_x + dx])))
            mean = np.array(vals).mean()
        except:
            mean = 255
        return int(mean)

    @classmethod
    def _get_beard_vals_region(cls, img_beard_gray, face_landmarks, vts):
        beard_vals = []
        for vt in vts:
            pt = cls.get_vt_coord(img_beard_gray, face_landmarks, vt)
            if pt is None:
                continue
            val = cls._get_beard_avg_val(img_beard_gray, pt)
            beard_vals.append(val)
        return beard_vals

    @classmethod
    def _has_beard_at_region(cls, bvals, val_has_beard):
        beard_cnt = 0
        for val in bvals:
            if val <= val_has_beard:
                beard_cnt += 1
        if beard_cnt >= 1:
            return True
        return False

    @classmethod
    def _get_box_img_around_mouth(cls, img_beard_gabor_gray, mesh_results):
        image_width, image_height = img_beard_gabor_gray.shape[0],  img_beard_gabor_gray.shape[1]
        num_vt1 = 206
        num_vt2 = 378

        face_landmarks = mesh_results.multi_face_landmarks[0]
        llm = face_landmarks.landmark

        rx1, ry1= llm[num_vt1].x, llm[num_vt1].y,
        px1, py1 = cls.normalized_to_pixel_coordinates(rx1, ry1,  image_width, image_height) 

        rx2, ry2= llm[num_vt2].x, llm[num_vt2].y,
        px2, py2 = cls.normalized_to_pixel_coordinates(rx2, ry2,  image_width, image_height) 

        img_mouth = img_beard_gabor_gray[py1:py2, px1:px2]
        return img_mouth

    @classmethod
    def _has_beard_at_keypoints(cls, img_beard_gabor_gray, mesh_results, img_box_mouth):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            return False 

        face_landmarks = mesh_results.multi_face_landmarks[0]

        mean_img = img_box_mouth.mean()
        if mean_img > 255 - 1:
            return False, 0

        bvals_ur = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [92, 165, 167])
        bvals_ul = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [393, 391, 322])
        bvals_mr = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [106, 182, 83])
        bvals_ml = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [313, 406, 335])
        bvals_br = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [211, 38, 208])
        bvals_bl = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [428, 262, 431])

        val_has_beard = int((255 + mean_img) / 2)
        if val_has_beard > 255 - 2:
            val_has_beard = 255 - 2

        hb_ur = cls._has_beard_at_region(bvals_ur, val_has_beard)
        hb_ul = cls._has_beard_at_region(bvals_ul, val_has_beard)
        hb_mr = cls._has_beard_at_region(bvals_mr, val_has_beard)
        hb_ml = cls._has_beard_at_region(bvals_ml, val_has_beard)
        hb_br = cls._has_beard_at_region(bvals_br, val_has_beard)
        hb_bl = cls._has_beard_at_region(bvals_bl, val_has_beard)

        if (hb_ur and hb_ul) or (hb_mr and hb_ml):
            return True, val_has_beard
        if (hb_br and hb_bl):
            if hb_ur or hb_ul or hb_mr or hb_ml:
                return True, val_has_beard
        return False, 0

    @classmethod
    def _clean_region(cls, image, mesh_results):
        img_beard_color = image.copy()
        cls.flood_fill(img_beard_color, pt_seed=(0,0), color_fill=FMB_SELFIE_FILL_COLOR)
  
        #img_beard_color = cls._clean_region_mouth_inner(img_beard_color, mesh_results)
        img_beard_color = cls._clean_region_mouth_outter(img_beard_color, mesh_results)
        img_beard_color = cls._clean_region_above_nose(img_beard_color, mesh_results)

        return img_beard_color

    @classmethod
    def _clean_region_mouth_inner(cls, image_color, mesh_results, color_fill=FMB_FILL_COLOR):
        FMB_PX_ALTER = {"U": (0.00, -0.015), "R":(0.015, 0.015), "L":(-0.015, -0.015), "B":(0.00, 0.015)}
        image_mouth_mask_inner, pt_mouth_mask_inner_seed = cls.get_beard_mouth_inner(image_color, mesh_results, FMB_PX_ALTER)
        img_beard_color = cv2.bitwise_and(image_color, image_color, mask = image_mouth_mask_inner)
        cls.flood_fill(img_beard_color, pt_seed=pt_mouth_mask_inner_seed, color_fill=color_fill)
        return img_beard_color

    @classmethod
    def _clean_region_mouth_outter(cls, image_color, mesh_results):
        FMB_PX_ALTER = {"U": (0, 0), "R":(0.05, 0.03), "L":(-0.05, 0.03), "B":(0.0, 0.1)}
        image_beard_mask_outter, pt_beard_mask_outter_seed = cls.get_beard_mask_outter(image_color, mesh_results, FMB_PX_ALTER)
        img_beard_color = cv2.bitwise_and(image_color, image_color, mask = image_beard_mask_outter)
        cls.flood_fill(img_beard_color, pt_seed=pt_beard_mask_outter_seed, color_fill=FMB_FILL_COLOR)
        return img_beard_color

    @classmethod
    def _clean_region_above_nose(cls, image_color, mesh_results):
        # clean up image above nose
        face_landmarks = mesh_results.multi_face_landmarks[0]
        pt_nose = cls.get_vt_coord(image_color, face_landmarks, 4)
        nose_y = pt_nose[1]
        image_witdh = image_color.shape[1]
        pts = [(0, 0), (image_witdh, 0), (image_witdh, nose_y), (0, nose_y)]
        cv2.fillConvexPoly(image_color, np.array(pts), FMB_FILL_COLOR)
        return image_color 

    @classmethod
    def _filter_by_gabor_by_degree(cls, image, theta_pi=0.5, lamda_div=2.0, ksize=5):
        #ksize = 5  # kernal size
        sigma = 1.0
        theta = np.pi * theta_pi # 90 degree
        lamda = np.pi / lamda_div # length of wave
        gamma = 0.5
        psi = 0

        # ksize = 3  # kernal size
        # sigma = 1.0
        # lamda = np.pi / 8  # length of wave
        # theta = np.pi * theta_pi # 90 degree
        # gamma = 0.5
        # psi = 0

        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        gabor_filters = kern 

        img_accum = np.zeros_like(image)
        for gabor_filter in gabor_filters:
            img_fimg = cv2.filter2D(image, cv2.CV_8UC1, gabor_filter)
            img_accum = np.maximum(img_accum, img_fimg, img_accum)
        return img_accum

    @classmethod
    def _filter_by_gabor_filter(cls, image):
        img_accum_1 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5, lamda_div=2.0, ksize=5)
        img_accum_2 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5, lamda_div=2.0, ksize=7)
        #img_accum_3 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5, lamda_div=8)
        #img_accum_4 = cls._filter_by_gabor_by_degree(image, theta_pi=0.75)

        img_accum = np.zeros(image.shape, dtype=np.float32)
        img_accum = img_accum_1.astype('float32')
        img_accum += img_accum_2.astype('float32')
        #img_accum += img_accum_3.astype('float32')
        #img_accum += img_accum_4.astype('float32')
        
        img_accum -= img_accum.min()
        img_accum /= img_accum.max()
        img_accum *= 255

        img_accum_uint8 = img_accum.astype("uint8")
        img_dst = cls.ipc_sharpen_color(img_accum_uint8)

        #img_dst = np.zeros_like(img_accum_uint8)
        #cv2.normalize(img_accum_uint8, img_dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img_dst

