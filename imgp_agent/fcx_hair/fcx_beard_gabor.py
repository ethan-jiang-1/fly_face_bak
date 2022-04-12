import numpy as np
import cv2

FMB_FILL_COLOR = (192, 192, 192)
FMB_SELFIE_FILL_COLOR = (192, 192, 192)
FMB_ENTOPY_FILL_VAL = 192

BBOX_ARONUD_MOUTH = (206, 410)
BBOX_ARONUD_BEARD = (137, 365)


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
        #img_beard_region_wb = cls.ipc_white_balance_color(img_beard_region)
        img_bread_region_nm = cls.ipc_normalize_color(img_beard_region)

        img_entopy_filtered = cls._get_entopy_filtered(img_bread_region_nm, mesh_results)
        img_gabor_filtered = cls._get_gabor_filtered(img_entopy_filtered, mesh_results)
        img_combine_result = cls._clean_region_mouth_inner(img_gabor_filtered, mesh_results, color_fill=(255,255,255), mode="none")

        img_combine_result_blur = cv2.blur(img_combine_result, (3, 3))
        img_combine_gray = cv2.cvtColor(img_combine_result_blur, cv2.COLOR_BGR2GRAY)
        
        img_box_mouth = cls._get_bbox_img_by_vetices(img_combine_gray, mesh_results)
        has_beard, val_has_beard = cls._has_beard_at_keypoints(img_combine_gray, mesh_results, img_box_mouth)
        img_beard_gray = cls._encode_beard_gray(img_combine_gray, has_beard, val_has_beard)
        #img_beard_gray_bbox = cls._get_beard_gray_box(img_beard_gray, mesh_results)
 
        if debug:
            from imgp_common import PlotHelper
            PlotHelper.plot_imgs([image, img_beard_region, img_bread_region_nm, img_entopy_filtered, img_gabor_filtered, img_combine_result, img_box_mouth, img_beard_gray], # , img_beard_gray_bbox],
                                 names=["org", "region", "normalized", "entopy", "gabor", "result", "mubox", "result", "result_bbox"])            

        #print(img_beard_white.shape, img_beard_white.dtype)
        return img_beard_gray

    @classmethod
    def _get_entopy_filtered(cls, img_color, mesh_results):
        img_color_et = cls._get_entopy_masked(img_color)
        img_color_et_sp = cls.ipc_sharpen_color(img_color_et)  
        #img_color_sp_nom = cls._clean_region_mouth_inner(img_color_et_sp, mesh_results, color_fill=(255,255,255), mode="enlarger")
        return img_color_et_sp    

    @classmethod
    def _get_gabor_filtered(cls, img_color, mesh_results):
        img_color_gb = cls._filter_by_gabor_filter(img_color)
        img_color_gb_nom = cls._clean_region_mouth_inner(img_color_gb, mesh_results, color_fill=(255,255,255), mode="enlarger")
        return img_color_gb_nom

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
    def _get_bbox_img_by_vetices(cls, img_beard_gray, mesh_results, vts=BBOX_ARONUD_MOUTH, extend_pt1=(0.0, 0.0), extend_pt2=(0.0,0.0)):
        image_width, image_height = img_beard_gray.shape[0],  img_beard_gray.shape[1]
        num_vt1 = vts[0]
        num_vt2 = vts[1]  # 378

        face_landmarks = mesh_results.multi_face_landmarks[0]
        llm = face_landmarks.landmark

        rx1, ry1= llm[num_vt1].x, llm[num_vt1].y,
        rx1 += extend_pt1[0]
        ry1 += extend_pt1[1]
        px1, py1 = cls.normalized_to_pixel_coordinates(rx1, ry1,  image_width, image_height) 

        rx2, ry2= llm[num_vt2].x, llm[num_vt2].y,
        rx2 += extend_pt2[0]
        ry2 += extend_pt2[1]
        px2, py2 = cls.normalized_to_pixel_coordinates(rx2, ry2,  image_width, image_height) 

        img_mouth = img_beard_gray[py1:py2, px1:px2]
        return img_mouth

    @classmethod
    def _has_beard_at_keypoints(cls, img_beard_gabor_gray, mesh_results, img_box_mouth):
        if mesh_results is None or mesh_results.multi_face_landmarks is None:
            return False 

        face_landmarks = mesh_results.multi_face_landmarks[0]

        mean_img = img_box_mouth.mean()
        val_has_beard = int((255 + mean_img) / 2)
        if val_has_beard > 255 - 2:
            val_has_beard = 255 - 2

        if mean_img > 255 - 1:
            return False, 0
        if mean_img < 245:
            return True, val_has_beard

        bvals_ur = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [92, 165, 167])
        bvals_ul = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [393, 391, 322])
        bvals_mr = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [106, 182, 83])
        bvals_ml = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [313, 406, 335])
        bvals_br = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [211, 38, 208])
        bvals_bl = cls._get_beard_vals_region(img_beard_gabor_gray, face_landmarks, [428, 262, 431])

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
  
        #img_beard_color = cls._clean_region_mouth_inner(img_beard_color, mesh_results, mode="fit")
        img_beard_color = cls._clean_region_beard_outter(img_beard_color, mesh_results, mode="enlarger")
        img_beard_color = cls._clean_region_above_nose(img_beard_color, mesh_results)

        return img_beard_color

    @classmethod
    def _clean_region_mouth_inner(cls, image_color, mesh_results, color_fill=FMB_FILL_COLOR, mode="fit"):
        if mode == "fit":
            FMB_PX_ALTER = {"U": (0.00, -0.000), "R":(0.002, 0.002), "L":(-0.002, -0.002), "B":(0.00, 0.002)}
        elif mode == "enlarger":
            FMB_PX_ALTER = {"U": (0.00, -0.004), "R":(0.008, 0.008), "L":(-0.008, -0.008), "B":(0.00, 0.016)}
        else:
            FMB_PX_ALTER = {"U": (0.00, -0.000), "R":(0.008, 0.000), "L":(-0.00, -0.00), "B":(0.00, 0.00)}
        image_mouth_mask_inner, pt_mouth_mask_inner_seed = cls.get_beard_mouth_inner(image_color, mesh_results, FMB_PX_ALTER)
        img_beard_color = cv2.bitwise_and(image_color, image_color, mask = image_mouth_mask_inner)
        cls.flood_fill(img_beard_color, pt_seed=pt_mouth_mask_inner_seed, color_fill=color_fill)
        return img_beard_color

    @classmethod
    def _clean_region_beard_outter(cls, image_color, mesh_results, color_fill=FMB_FILL_COLOR, mode="fit"):
        if mode == "fit":
            FMB_PX_ALTER = {"U": (0.00, -0.002), "R":(0.004, 0.004), "L":(-0.004, -0.004), "B":(0.00, 0.004)}
        elif mode == "enlarger":
            FMB_PX_ALTER = {"U": (0.00, 0.00), "R":(0.05, 0.03), "L":(-0.05, 0.03), "B":(0.00, 0.10)}
        else:
            FMB_PX_ALTER = {"U": (0.00, -0.00), "R":(0.00, 0.00), "L":(-0.00, -0.00), "B":(0.00, 0.00)}

        image_beard_mask_outter, pt_beard_mask_outter_seed = cls.get_beard_mask_outter(image_color, mesh_results, FMB_PX_ALTER)
        img_beard_color = cv2.bitwise_and(image_color, image_color, mask = image_beard_mask_outter)
        cls.flood_fill(img_beard_color, pt_seed=pt_beard_mask_outter_seed, color_fill=color_fill)
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
    def _combine_all_imgs(cls, imgs):
        img_accum = np.zeros(imgs[0].shape, dtype=np.float32)

        for img in imgs:
            img_accum += img.astype('float32')
        
        img_accum -= img_accum.min()
        img_accum /= img_accum.max()
        img_accum *= 255

        img_accum_uint8 = img_accum.astype("uint8")
        return img_accum_uint8

    @classmethod
    def _filter_by_gabor_filter(cls, image):
        img_accum_1 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5, lamda_div=2.0, ksize=5)
        #img_accum_2 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5, lamda_div=2.0, ksize=7)
        #img_accum_3 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5+0.02, lamda_div=2.0, ksize=5)
        #img_accum_4 = cls._filter_by_gabor_by_degree(image, theta_pi=0.5-0.02, lamda_div=2.0, ksize=5)

        img_accum_uint8 = cls._combine_all_imgs([img_accum_1]) # , img_accum_2]) # , img_accum_4])
        img_dst = cls.ipc_sharpen_color(img_accum_uint8)

        #img_dst = np.zeros_like(img_accum_uint8)
        #cv2.normalize(img_accum_uint8, img_dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return img_dst

    @classmethod
    def _get_entopy_masked_core(cls, img_org, disk_num=8, th_val=FMB_ENTOPY_FILL_VAL, fill_gray=FMB_ENTOPY_FILL_VAL):
        from skimage.filters.rank import entropy
        from skimage.morphology import disk

        img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)

        img_entr = entropy(img_gray, disk(disk_num))

        img_mask = img_entr.copy()
        img_mask -= img_mask.min()
        img_mask /= img_mask.max()
        img_mask *= 255

        img_mask_u8 = img_mask.astype(np.uint8)
        _, img_mask_u8 = cv2.threshold(img_mask_u8, th_val, 255, cv2.THRESH_BINARY)

        img_mask_mono = img_mask_u8.copy()
        img_mask_mono[np.where(img_mask_mono[:,:] == 0)] = fill_gray
    
        img_masked_filled = cv2.bitwise_and(img_org, img_org, mask = img_mask_u8)   
        img_mask_color = cv2.cvtColor(img_mask_mono, cv2.COLOR_GRAY2RGB)
        img_masked_filled += img_mask_color
        return img_masked_filled

    @classmethod
    def _get_entopy_masked(cls, img_org):
        #img_et1 = cls._get_entopy_masked_core(img_org, disk_num=1)
        img_et2 = cls._get_entopy_masked_core(img_org, disk_num=2)
        #img_et4 = cls._get_entopy_masked_core(img_org, disk_num=4)
        #img_et8 = cls._get_entopy_masked_core(img_org, disk_num=8)

        img_accum_uint8 = cls._combine_all_imgs([img_et2]) # , img_et4, img_et8])
        #img_accum_uint8 = cls.ipc_sharpen_color(img_accum_uint8)
        return img_accum_uint8

    @classmethod
    def _encode_beard_gray(cls, img_combine_gray, has_beard, val_has_beard):
        img_gray_float = img_combine_gray.astype('float32')
        img_gray_float -= img_gray_float.min()
        img_gray_float /= img_gray_float.max()

        if not has_beard:
            img_gray_float += 3 * img_gray_float.max()
            img_gray_float *= 255 / 4
        else:
            img_gray_float *= 255

        img_gray_uint8 = img_gray_float.astype("uint8")
        return img_gray_uint8

    @classmethod
    def _get_beard_gray_box(cls, img_beard_gray, mesh_results):
        extend_pt1 = (-0.02, 0.0)
        extend_pt2 = (0.08, 0.10)
        img_bbox_gray = cls._get_bbox_img_by_vetices(img_beard_gray, mesh_results, vts=BBOX_ARONUD_BEARD, extend_pt1=extend_pt1, extend_pt2=extend_pt2)
        img_resized = cv2.resize(img_bbox_gray.copy(), (512, 512))
        return img_resized

        
